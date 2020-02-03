#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_processing/screen_points.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Point32.h>

static const std::string OPENCV_WINDOW = "Open-CV display window";

using namespace std;

/*
*   TODOS :
*   1.add threshold command, outputQuad 
*   2.all commands-> 1 Subscriber
*   3.Switch msgs to more effective ones
*   4.A more info statements
*/


class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Subscriber point_sub_;
  ros::Subscriber command_sub_;
  cv::Point2f outputQuad[4];
  cv::Point2f inputQuad[4];
  cv::Mat screenshot;
  bool take_image;
  bool pub_positions;
  image_transport::Publisher image_pub_;
  ros::Publisher position_pub_;
public:


  ImageConverter(ros::NodeHandle &nodehandle)
  : it_(nh_) {
      // Subscribe to input video feed and publish output video feed
      image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &ImageConverter::imageCb, this);
      point_sub_ = nh_.subscribe("/image_processing/screen_setup", 1, &ImageConverter::points_setup, this);
      command_sub_ = nh_.subscribe("/image_processing/command", 1, &ImageConverter::command, this );
      image_pub_ = it_.advertise("/image_processing/debug", 1);
      position_pub_ = nh_.advertise<geometry_msgs::Point32>("/image_processing/position", 1);
      // The 4 points that select quadilateral on the input , from top-left in clockwise order
      // These four pts are the sides of the rect box used as input
      inputQuad[0] = cv::Point2f(0,0);
      inputQuad[1] = cv::Point2f(640,0);
      inputQuad[2] = cv::Point2f(640,480);
      inputQuad[3] = cv::Point2f(0,480);
      // The 4 points where the mapping is to be done , from top-left in clockwise order
      outputQuad[0] = cv::Point2f(0,0);
      outputQuad[1] = cv::Point2f(400,0);
      outputQuad[2] = cv::Point2f(400,400);
      outputQuad[3] = cv::Point2f(0,400);
      take_image = true;

      cv::namedWindow(OPENCV_WINDOW);
  }

  ~ImageConverter() {
      cv::destroyWindow(OPENCV_WINDOW);
  }
  //image comes in as a ROS message, but gets converted to an OpenCV type
  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr; //OpenCV data type
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat gray_image, lambda, output, aux;
    //convert the color image to grayscale:
    cv::cvtColor(cv_ptr->image, gray_image, CV_BGR2GRAY);
    // Set the lambda matrix the same type and size as input
    lambda = cv::Mat::zeros( gray_image.rows, gray_image.cols, gray_image.type() );
    // Get the Perspective Transform Matrix i.e. lambda
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(gray_image,output,lambda, cv::Size(400, 400) );
    /// Reduce noise with a kernel 3x3
    blur( output, output, cv::Size(3,3) );
    // Convert image to float32
    output.convertTo(output, CV_32F);
    // take a picture if no picture exists or needs update
    if(this->take_image)
    {
      this->screenshot = output;
      ROS_INFO("New Screenshot taken!\n");

      this->take_image = false;
    }
    // divide the image by the pattern
    aux = 1-(output/this->screenshot);
    // convert back to
    aux.convertTo(aux, CV_8U, 255);

    // debug code
    //cv_ptr->image = this->screenshot;
    //cv_ptr->encoding = "mono8";
    //image_pub_.publish(cv_ptr->toImageMsg());

    // apply threshold TODO: being able to change threshold with a command
    cv::threshold(aux, aux, 30, 255, 0);

    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    // Find contours
    cv::findContours( aux, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    if(contours.size() > 0)
    {
      /// Get the moments
      vector<cv::Moments> mu(contours.size() );
      for( int i = 0; i < contours.size(); i++ )
      {
        mu[i] = moments( contours[i], false );
      }
      ///  Get the mass centers:
      vector<cv::Point2f> mc( contours.size() );
      for( int i = 0; i < contours.size(); i++ )
      {
        mc[i] = cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
      }
        // publish the centers of mass
      if(this->pub_positions == true)
      {
        ROS_INFO("%lu object(s) detected!\n", contours.size());
        for( int i = 0; i < contours.size(); i++ )
        {
          geometry_msgs::Point32 pmsg;
          pmsg.x = mc[i].x;
          pmsg.y = mc[i].y;
          pmsg.z = 1;
          position_pub_.publish(pmsg);
        }
        this->pub_positions = false;
      }


      // drawContours
      cv::Mat drawing = cv::Mat::zeros( aux.size(), CV_8UC3 );
      for( int i = 0; i< contours.size(); i++ )
      {
        cv::Scalar color = cv::Scalar( 0, 0, 255 );
        cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
        cv::circle( drawing, mc[i], 4, color, -1, 8, 0 );
      }
      cv::imshow(OPENCV_WINDOW, drawing);
      cv::waitKey(3); //need waitKey call to update OpenCV image window


    }else
    {
      cv::imshow(OPENCV_WINDOW, aux);
      cv::waitKey(3); //need waitKey call to update OpenCV image window
    }
  }

  void points_setup(const image_processing::screen_points& mesg)
  {
    inputQuad[0] = cv::Point2f(mesg.points[0], mesg.points[1]);
    inputQuad[1] = cv::Point2f(mesg.points[2], mesg.points[3]);
    inputQuad[2] = cv::Point2f(mesg.points[4], mesg.points[5]);
    inputQuad[3] = cv::Point2f(mesg.points[6], mesg.points[7]);
    take_image = true;
    ROS_INFO("Image points updated to:\nUpLeft    : ( %f,%f )\nUpRight   : ( %f,%f )\nDownRight : ( %f,%f )\nDownLeft  : ( %f,%f )\n", inputQuad[0].x, inputQuad[0].y, inputQuad[1].x, inputQuad[1].y, inputQuad[2].x, inputQuad[2].y, inputQuad[3].x ,inputQuad[3].y);
  }

  void command(const std_msgs::String& com)
  {
    if(com.data == "SCREENSHOT")
    {
      this->take_image = true;
    }else if(com.data == "POSITIONS")
    {
      this->pub_positions = true;
    }
  }

}; //end of class definition


int main(int argc, char** argv) {
  ros::init(argc, argv, "image_processing");
  ros::NodeHandle n; //
  ImageConverter ic(n); // instantiate object of class ImageConverter
  ros::Duration timer(0.1);
  while (ros::ok()) {
      ros::spinOnce();
      timer.sleep();
  }
return 0;
}
