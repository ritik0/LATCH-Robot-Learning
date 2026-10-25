#include <algorithm>
#include <cmath>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class SafetyFilter : public rclcpp::Node
{
    public:
        SafetyFilter()
        :   Node("safety_filter")
        {
            publisher_ = this->create_publisher<
            std_msgs::msg::Float64MultiArray>(
                "/latch/action",
                10
            );

            subscription_ = this->create_subscription<
            std_msgs::msg::Float64MultiArray>(
                "/latch/policy_action",
                10,
                std::bind(
                    &SafetyFilter::action_callback,
                    this,
                    std::placeholders::_1
                )
            );

            RCLCPP_INFO(
                this->get_logger(),
                "Safety filter started"
            );
        }

    private:
        void action_callback(
            const std_msgs::msg::Float64MultiArray::SharedPtr msg)
            {
                if(msg->data.size() != 4)
                {
                    RCLCPP_WARN(
                        this->get_logger(),
                        "Rejected action: expected 4 values, got %zu",
                        msg->data.size()
                    );

                    return;
                }

                std_msgs::msg::Float64MultiArray safe_msg;
                safe_msg.data.resize(4);

                for (std::size_t i = 0; i < 4; ++i)
                {
                    double value = msg->data[i];
                    if (!std::isfinite(value))
                    {
                        RCLCPP_WARN(
                            this->get_logger(),
                            "Rejected non-finite action");

                        return;
                    }
                    safe_msg.data[i] = std::clamp(
                        value,
                        -1.0,
                        1.0
                    );
                }

                publisher_->publish(
                    safe_msg
                );
            }
        rclcpp::Publisher<
            std_msgs::msg::Float64MultiArray
        >::SharedPtr publisher_;

        rclcpp::Subscription<
            std_msgs::msg::Float64MultiArray
        >::SharedPtr subscription_;

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SafetyFilter>();

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}