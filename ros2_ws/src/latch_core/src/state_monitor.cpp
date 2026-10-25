#include <memory>
#include <functional>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class StateMonitor : public rclcpp::Node
{
    public:
        StateMonitor()
        : Node("state_monitor")
        {
            subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
                "/latch/observation",
                10,
                std::bind(
                    &StateMonitor::observation_callback,
                    this,
                    std::placeholders::_1
                )
            );

            RCLCPP_INFO(
                this->get_logger(),
                "State monitor started"
            );
        }
    private:
        void observation_callback(
            const std_msgs::msg::Float64MultiArray::SharedPtr msg)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "Received observation with %zu values",
                msg->data.size()
            );
        }

        rclcpp::Subscription<
            std_msgs::msg::Float64MultiArray
        >::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StateMonitor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}