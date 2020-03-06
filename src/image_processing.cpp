#include <ros/ros.h>
#include <string>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_processing/PassValue.h>
#include <image_processing/Point2D.h>
#include <image_processing/Point2DArray.h>
#include <image_processing/Point2DStamped.h>
#include <image_processing/Point2DStampedArray.h>
#include <image_processing/Size.h>
#include <std_srvs/Empty.h>



static const std::string OPENCV_WINDOW = "Center Masses Image";
static const std::string OPENCV_WINDOW_TR = "Transformed/Blured Image";
static const std::string OPENCV_WINDOW_THR = "Threshold-ed Image";
static const std::string OPENCV_WINDOW_OR = "Original Image";

using namespace std;

/*
*   TODOS :
*   GUI for threshold /
*   https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html
*/


class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  ros::Publisher position_pub_;
  cv::Point2f outputQuad[4];
  cv::Point2f inputQuad[4];
  cv::Mat screenshot;
  bool take_image;
  bool pub_positions;
  int threshold;
  int output_image_x;
  int output_image_y;

public:


  ImageConverter(ros::NodeHandle &nodehandle)
  : it_(nh_) {
      // Subscribe to input video feed and publish output video feed
      image_sub_ = it_.subscribe("/usb_cam/image_raw", 1, &ImageConverter::imageCb, this);

      // Position publisher
      position_pub_ = nh_.advertise<image_processing::Point2DStampedArray>("/image_processing/positions", 1);

      // Iniatial values for given variables
      take_image = true;
      pub_positions = false;
      threshold = 30;
      output_image_x = 520; 
      output_image_y = 320; 

      // The 4 points that select quadilateral on the input , from top-left in clockwise order
      // These four pts are the sides of the rect box used as input
      inputQuad[0] = cv::Point2f(0,0);
      inputQuad[1] = cv::Point2f(640,0);
      inputQuad[2] = cv::Point2f(640,480);
      inputQuad[3] = cv::Point2f(0,480);
      // The 4 points where the mapping is to be done , from top-left in clockwise order
      outputQuad[0] = cv::Point2f(0,0);
      outputQuad[1] = cv::Point2f(output_image_x,0);
      outputQuad[2] = cv::Point2f(output_image_x,output_image_y);
      outputQuad[3] = cv::Point2f(0,output_image_y);

      // Debuggin and demonstration windows
      cv::namedWindow(OPENCV_WINDOW_THR);
      cv::namedWindow(OPENCV_WINDOW_TR);
      cv::namedWindow(OPENCV_WINDOW);
      cv::namedWindow(OPENCV_WINDOW_OR);
      // placing the windows far appart
      cv::moveWindow(OPENCV_WINDOW_TR, 640,0);
      cv::moveWindow(OPENCV_WINDOW_THR, 640+output_image_x,0);
      cv::moveWindow(OPENCV_WINDOW, 0,480);

  }

  ~ImageConverter() {
      cv::destroyWindow(OPENCV_WINDOW);
      cv::destroyWindow(OPENCV_WINDOW_OR);
      cv::destroyWindow(OPENCV_WINDOW_THR);
      cv::destroyWindow(OPENCV_WINDOW_TR);
  }

  /******************************** imageCb ************************************
  *  This function is the image conversion loop
  *
  *****************************************************************************/
  void imageCb(const sensor_msgs::ImageConstPtr& msg) {
    //image comes in as a ROS message, but gets converted to an OpenCV type
    cv_bridge::CvImagePtr cv_ptr; //OpenCV data type
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    // Show image for debugging purposes
    cv::imshow(OPENCV_WINDOW_OR, cv_ptr->image);
    cv::waitKey(3); //need waitKey call to update OpenCV image window
    cv::Mat gray_image, lambda, output, aux;
    //convert the color image to grayscale:
    cv::cvtColor(cv_ptr->image, gray_image, CV_BGR2GRAY);
    // Set the lambda matrix the same type and size as input
    lambda = cv::Mat::zeros( gray_image.rows, gray_image.cols, gray_image.type() );
    // Get the Perspective Transform Matrix i.e. lambda
    lambda = getPerspectiveTransform( inputQuad, outputQuad );
    // Apply the Perspective Transform just found to the src image
    warpPerspective(gray_image,output,lambda, cv::Size(this->output_image_x, this->output_image_y) );
    /// Reduce noise with a kernel 3x3
    blur( output, output, cv::Size(3,3) );
    // Show image for debugging purposes
    cv::imshow(OPENCV_WINDOW_TR, output);
    cv::waitKey(3); //need waitKey call to update OpenCV image window
    // Convert image to float32
    output.convertTo(output, CV_32F);
    // take a picture if no picture exists or needs update
    if(this->take_image)
    {
      this->screenshot = output;
      ROS_INFO("New Screenshot taken!");

      this->take_image = false;
    }
    // divide the image by the pattern
    aux = 1-(output/this->screenshot);
    // convert back to
    aux.convertTo(aux, CV_8U, 255);

    // apply threshold 
    cv::threshold(aux, aux, this->threshold, 255, 0);
    // Show image for debugging purposes
    cv::imshow(OPENCV_WINDOW_THR, aux);
    cv::waitKey(3); //need waitKey call to update OpenCV image window
    // Vectors need to store contours and their heirarchy
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy; // needed for cv::findContours but not needed by the program
    // Find contours
    cv::findContours( aux, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

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
        ROS_INFO("%lu object(s) detected!", contours.size());
        image_processing::Point2DStampedArray msg_arr;
        for( int i = 0; i < contours.size(); i++ )
        {
          image_processing::Point2DStamped pmsg;
          pmsg.id = i+1;
          pmsg.point.x = mc[i].x;
          pmsg.point.y = mc[i].y;
          msg_arr.points.push_back(pmsg);
        }
        position_pub_.publish(msg_arr);
        this->pub_positions = false;
      }
      // drawContours
      cv::Mat drawing = cv::Mat::zeros( aux.size(), CV_8UC3 );
      for( int i = 0; i< contours.size(); i++ )
      {
        cv::Scalar color = cv::Scalar( 0, 0, 255 );
        cv::drawContours( drawing, contours, i, color, 1);
        cv::circle( drawing, mc[i], 1, color, -1, 8, 0 );
      }
      cv::imshow(OPENCV_WINDOW, drawing);
      cv::waitKey(3); //need waitKey call to update OpenCV image window
    }else
    {
      if(this->pub_positions == true)
      {
        ROS_INFO("No object(s) detected!");
        this->pub_positions = false;
      }
    }
  }

  /**************************** threshold_setup ********************************
  *  This service sets the threshold.
  *
  *  Example Usage :
  *  rosservice call /threshold_setup "x: 50"
  *****************************************************************************/
  bool threshold_setup(image_processing::PassValue::Request &req,
                      image_processing::PassValue::Response &res)
  {
    this->threshold = req.x;
    ROS_INFO("New threshold: %d", this->threshold);
    return true;
  }

  /******************************* points_setup ********************************
  *  This service takes the points that make up our working surface and saves them in an array.
  *
  *  Example Usage :
  *  rosservice call /image_processing/setup/screen_setup "{points:[{x: 35, y: 54}, {x: 424, y: 22}, {x: 471, y: 414}, {x: 60, y: 456}]}"
  *****************************************************************************/
  bool points_setup(image_processing::Point2DArray::Request &req, image_processing::Point2DArray::Response &res)
  {
    int i = 0;
    for(vector<image_processing::Point2D>::const_iterator point = req.points.begin(); point != req.points.end(); ++point)
    {
      inputQuad[i] = cv::Point2f(point->x, point->y);
      i++;
    }
    take_image = true;
    ROS_INFO("Image points updated to:\n\tUpLeft    : ( x: %.0f \ty: %.0f )\n\tUpRight   : ( x: %.0f \ty: %.0f )\n\tDownRight : ( x: %.0f \ty: %.0f )\n\tDownLeft  : ( x: %.0f \ty: %.0f )\n",
                                        inputQuad[0].x, inputQuad[0].y,
                                        inputQuad[1].x, inputQuad[1].y,
                                        inputQuad[2].x, inputQuad[2].y,
                                        inputQuad[3].x ,inputQuad[3].y);
    return true;
  }

    /******************************* size_setup ********************************
  *  This service changes the output image size to x columns,y rows
  *
  *  Example Usage :
  *  rosservice call /image_processing/setup/size "x: 520 y: 320"
  *****************************************************************************/
  bool size_setup(image_processing::Size::Request &req, image_processing::Size::Response &res)
  {
    this->output_image_x = req.x;
    this->output_image_y = req.y;
    this->outputQuad[1] = cv::Point2f(this->output_image_x,0);
    this->outputQuad[2] = cv::Point2f(this->output_image_x,this->output_image_y);
    this->outputQuad[3] = cv::Point2f(0,this->output_image_y);
    this->take_image = true;
    ROS_INFO("Output image size updated to - x: %d y: %d", req.x, req.y);
    return true;
  }

  /********************************* command ***********************************
  *  This function is used in order to take a screenshot pattern or to publish
  *  the object positions to /image_processing/position.
  *
  *  Example Usage :
  *  rosservice call /image_processing/publish "{}"
  *****************************************************************************/
  bool points_pub(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
  {
      ROS_INFO("Publishing the object's position");
      this->pub_positions = true;
      return true;
  }
}; //end of class definition


int main(int argc, char** argv) {
  ros::init(argc, argv, "image_processing");
  ros::NodeHandle n; //
  ImageConverter ic(n); // instantiate object of class ImageConverter
  
  // Services needed for setup
  ros::ServiceServer threshhold_ = n.advertiseService("/image_processing/setup/threshold", &ImageConverter::threshold_setup, &ic);
  ros::ServiceServer output_size_ = n.advertiseService("/image_processing/setup/size", &ImageConverter::size_setup, &ic);
  ros::ServiceServer publish_ = n.advertiseService("/image_processing/publish", &ImageConverter::points_pub, &ic);
  ros::ServiceServer point_sub_ = n.advertiseService("/image_processing/setup/screen_setup", &ImageConverter::points_setup, &ic);

  ros::Duration timer(0.1); // our loop runs 10 times a second
  while (ros::ok()) {
      ros::spinOnce();
      timer.sleep();
  }
return 0;
}
