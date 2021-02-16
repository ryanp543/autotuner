#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
//#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <string>

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

      // Find out the name set for namespace_model in the definition file (if it exists)
      if (_sdf->HasElement("namespace_model"))
          this->namespace_model = _sdf->Get<std::string>("namespace_model");
      if (_sdf->HasElement("number_of_wheels"))
          this->number_of_wheels = _sdf->Get<int>("number_of_wheels");
      if (_sdf->HasElement("wheel_radius"))
          this->wheel_radius = _sdf->Get<double>("wheel_radius");

      // Create ROS topics names to subscribe to
      //std::string left_wheel_speed = "/"+this->namespace_model + "/left_wheel_speed";
      //std::string right_wheel_speed = "/"+this->namespace_model + "/right_wheel_speed";
      
      // Create ROS topic names to publish
      // MAYBE for the reaction forces

      // Initialize ros, if it has not already bee initialized.
      //if (!ros::isInitialized())
      //{
      //  int argc = 0;
      //  char **argv = NULL;
      //  ros::init(argc, argv, "set_wheelfriction_rosnode",
      //      ros::init_options::NoSigintHandler);
      //}
         
      // Create our ROS node. This acts in a similar manner to
      // the Gazebo node
      //this->rosNode.reset(new ros::NodeHandle("set_wheelfriction_rosnode"));
      
      // Enable the feedback of internal forces in the wheel joints
      for (int i = 1; i < number_of_wheels+1; i++)
      {
        this->model->GetJoint("joint_wheel"+std::to_string(i))->SetProvideFeedback(1);
      }
     

      // Freq
      //ros::SubscribeOptions so =
      //  ros::SubscribeOptions::create<std_msgs::Float32>(
      //      left_wheel_speed,
      //      1,
      //      boost::bind(&ModelJointControler::OnRosMsg_left_wheel_speed, this, _1),
      //      ros::VoidPtr(), &this->rosQueue);
      //this->rosSub = this->rosNode->subscribe(so);
      
      // Spin up the queue helper thread.
      //this->rosQueueThread =
      //  std::thread(std::bind(&ModelJointControler::QueueThread, this));
        
        
      // Magnitude
      //ros::SubscribeOptions so2 =
      //  ros::SubscribeOptions::create<std_msgs::Float32>(
      //      right_wheel_speed,
      //      1,
      //      boost::bind(&ModelJointControler::OnRosMsg_right_wheel_speed, this, _1),
      //      ros::VoidPtr(), &this->rosQueue2);
      //this->rosSub2 = this->rosNode->subscribe(so2);
      
      // Spin up the queue helper thread.
      //this->rosQueueThread2 =
      //  std::thread(std::bind(&ModelJointControler::QueueThread2, this));
         
      ROS_INFO("Loaded Custom Friction Plugin with parent...%s", this->model->GetName().c_str());
      ROS_INFO("     Number of wheels: %i", number_of_wheels);
      ROS_INFO("     Radius of the wheels [m]: %f", wheel_radius);
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      double new_secs =ros::Time::now().toSec();
      double delta = new_secs - this->old_secs;
      
      double max_delta = 0.0;
      
      // Check if delta time is larger than max_delta
      if (this->freq_update != 0.0)
      {
        max_delta = 1.0 / this->freq_update;
      }
      
      if (delta > max_delta && delta != 0.0)
      {
        this->old_secs = new_secs;
        //ROS_INFO("New simulation iteration detected.");

        // Add friction to each wheel
        for (int i = 1; i<number_of_wheels+1; i++){ AddFrictionToWheel(i); }

      }

    } 
    
    public: void AddFrictionToWheel(const int wheel_index)
    {
       // Get the contact forces applied at the wheel
        ignition::math::Vector3<double> joint_force = this->model->GetJoint("joint_wheel"+std::to_string(wheel_index))->LinkForce(0); // Force due to wheel joint in world reference frame
        ignition::math::Vector3<double> relative_contact_position = this->wheel_radius*joint_force.Normalize(); // Position of contact point wrt wheel center in world ref frame
        // Get the position in the body reference frame where to apply the traction force
        ignition::math::Vector3<double> global_contact_position = relative_contact_position+this->model->GetLink("link_wheel"+std::to_string(wheel_index))->WorldCoGPose().Pos();


        // Get the relative velocity of the wheels WRT average contact point
        ignition::math::Vector3<double> contactpoint_velocity = this->model->GetLink("link_wheel"+std::to_string(wheel_index))->WorldLinearVel(relative_contact_position,ignition::math::Quaternion<double>(0,0,0));
        std::cout << "Contact point velocity [m/s]: " << contactpoint_velocity << std::endl;
        // Adjusting the relative velocity to enforce contact condition
        double normal_component = contactpoint_velocity.Dot(joint_force.Normalize());
        ignition::math::Vector3<double>  contactpoint_velocity_corrected = contactpoint_velocity-normal_component*joint_force.Normalize();

        // Apply desired forces to the model joints.
        ROS_INFO("Update friction force at wheel %i", wheel_index );
        ROS_INFO("Relative speed [m/s]: %f", contactpoint_velocity_corrected.Length());
        ignition::math::Vector3<double> force = -0.9*joint_force.Length()*contactpoint_velocity/(contactpoint_velocity.Length()+0.01);
        std::cout << "Force appliquee [N]: " << force << std::endl;
        //this->model->GetLink("link_chassis")->AddForceAtWorldPosition(force,global_contact_position);
      
    }

    /*
    public: void SetLeftWheelSpeed(const double &_freq)
    {
      this->left_wheel_speed_magn = _freq;
      //ROS_INFO("left_wheel_speed_magn >> %f", this->left_wheel_speed_magn);
    }
    
    public: void SetRightWheelSpeed(const double &_magn)
    {
      this->right_wheel_speed_magn = _magn;
      //ROS_INFO("right_wheel_speed_magn >> %f", this->right_wheel_speed_magn);
    }
    
    
    public: void OnRosMsg_left_wheel_speed(const std_msgs::Float32ConstPtr &_msg)
    {
      this->SetLeftWheelSpeed(_msg->data);
    }
    */
    
    /// \brief ROS helper function that processes messages
    private: void QueueThread()
    {
      static const double timeout = 0.01;
      while (this->rosNode->ok())
      {
        this->rosQueue.callAvailable(ros::WallDuration(timeout));
      }
    }
    
    /*
    public: void OnRosMsg_right_wheel_speed(const std_msgs::Float32ConstPtr &_msg)
    {
      this->SetRightWheelSpeed(_msg->data);
    }
    */
    
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
    
    // Max. Update Frequency
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

    std::string namespace_model = "";
    int number_of_wheels;
    double wheel_radius;

  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(ModelJointControler)
}
