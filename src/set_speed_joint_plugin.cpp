#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>

namespace gazebo
{
  class ModelJointControler : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
    {
      // Store the pointer to the model
      this->model = _parent;

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ModelJointControler::OnUpdate, this));
      
      this->old_secs =ros::Time::now().toSec();

      if (_sdf->HasElement("namespace_model"))
          this->namespace_model = _sdf->Get<std::string>("namespace_model");

      // Create a topic name
      std::string left_wheel_speed = "/"+this->namespace_model + "/left_wheel_speed";
      std::string right_wheel_speed = "/"+this->namespace_model + "/right_wheel_speed";


      // Initialize ros, if it has not already bee initialized.
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "set_wheelSpeed_rosnode",
            ros::init_options::NoSigintHandler);
      }
         
      // Create our ROS node. This acts in a similar manner to
      // the Gazebo node
      this->rosNode.reset(new ros::NodeHandle("earthquake_rosnode"));
      

      // Freq
      //ROS_WARN("TEST 1.");
      ros::SubscribeOptions so =
        ros::SubscribeOptions::create<std_msgs::Float32>(
            left_wheel_speed,
            1,
            boost::bind(&ModelJointControler::OnRosMsg_left_wheel_speed, this, _1),
            ros::VoidPtr(), &this->rosQueue);
      this->rosSub = this->rosNode->subscribe(so);
      
      // Spin up the queue helper thread.
      this->rosQueueThread =
        std::thread(std::bind(&ModelJointControler::QueueThread, this));
        
        
      // Magnitude
      //ROS_WARN("TEST 2.");
      ros::SubscribeOptions so2 =
        ros::SubscribeOptions::create<std_msgs::Float32>(
            right_wheel_speed,
            1,
            boost::bind(&ModelJointControler::OnRosMsg_right_wheel_speed, this, _1),
            ros::VoidPtr(), &this->rosQueue2);
      this->rosSub2 = this->rosNode->subscribe(so2);
      
      // Spin up the queue helper thread.
      this->rosQueueThread2 =
        std::thread(std::bind(&ModelJointControler::QueueThread2, this));
         
      ROS_WARN("Loaded Plugin with parent...%s", this->model->GetName().c_str());
      
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      double new_secs =ros::Time::now().toSec();
      double delta = new_secs - this->old_secs;
      
      double max_delta = 0.0;
      
      if (this->freq_update != 0.0)
      {
        max_delta = 1.0 / this->freq_update;
      }
      
      if (delta > max_delta && delta != 0.0)
      {
        this->old_secs = new_secs;
        ROS_WARN("New simulation iteration detected.");

        // Apply a small linear velocity to the model.
        ROS_DEBUG("Update Wheel Speed BASIC...");
        this->model->GetJoint("joint_wheel1")->SetVelocity(0, this->left_wheel_speed_magn); 
        this->model->GetJoint("joint_wheel2")->SetVelocity(0, this->left_wheel_speed_magn); 
        this->model->GetJoint("joint_wheel3")->SetVelocity(0, this->left_wheel_speed_magn); 
        this->model->GetJoint("joint_wheel4")->SetVelocity(0, this->left_wheel_speed_magn); 
        this->model->GetJoint("joint_wheel5")->SetVelocity(0, this->right_wheel_speed_magn);
        this->model->GetJoint("joint_wheel6")->SetVelocity(0, this->right_wheel_speed_magn);
        this->model->GetJoint("joint_wheel7")->SetVelocity(0, this->right_wheel_speed_magn);
        this->model->GetJoint("joint_wheel8")->SetVelocity(0, this->right_wheel_speed_magn);
        
      }

    }
    
    
    public: void SetLeftWheelSpeed(const double &_freq)
    {
      this->left_wheel_speed_magn = _freq;
      //ROS_WARN("left_wheel_speed_magn >> %f", this->left_wheel_speed_magn);
    }
    
    public: void SetRightWheelSpeed(const double &_magn)
    {
      this->right_wheel_speed_magn = _magn;
      //ROS_WARN("right_wheel_speed_magn >> %f", this->right_wheel_speed_magn);
    }
    
    
    public: void OnRosMsg_left_wheel_speed(const std_msgs::Float32ConstPtr &_msg)
    {
      this->SetLeftWheelSpeed(_msg->data);
    }
    
    /// \brief ROS helper function that processes messages
    private: void QueueThread()
    {
      static const double timeout = 0.01;
      while (this->rosNode->ok())
      {
        this->rosQueue.callAvailable(ros::WallDuration(timeout));
      }
    }
    
    public: void OnRosMsg_right_wheel_speed(const std_msgs::Float32ConstPtr &_msg)
    {
      this->SetRightWheelSpeed(_msg->data);
    }
    
    /// \brief ROS helper function that processes messages
    private: void QueueThread2()
    {
      static const double timeout = 0.01;
      while (this->rosNode->ok())
      {
        this->rosQueue2.callAvailable(ros::WallDuration(timeout));
      }
    }
    

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
    
    // Time Memory
    double old_secs;
    
    // Frequency of earthquake
    double freq_update = 10.0;

    double left_wheel_speed_magn = 0.0;
    // Magnitude of the Oscilations
    double right_wheel_speed_magn = 0.0;

    
    
    /// \brief A node use for ROS transport
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    
    /// \brief A ROS subscriber
    private: ros::Subscriber rosSub;
    /// \brief A ROS callbackqueue that helps process messages
    private: ros::CallbackQueue rosQueue;
    /// \brief A thread the keeps running the rosQueue
    private: std::thread rosQueueThread;
    
    
    /// \brief A ROS subscriber
    private: ros::Subscriber rosSub2;
    /// \brief A ROS callbackqueue that helps process messages
    private: ros::CallbackQueue rosQueue2;
    /// \brief A thread the keeps running the rosQueue
    private: std::thread rosQueueThread2;
    
    std::string right_wheel_name;
    std::string left_wheel_name;

    std::string namespace_model = "";

    
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelJointControler)
}
