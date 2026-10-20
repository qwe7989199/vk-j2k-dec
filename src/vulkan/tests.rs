#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_api_version() {
        assert_eq!(format_api_version(make_api_version(1, 4, 303)), "1.4.303");
    }

    #[test]
    fn picks_compute_only_queue_before_graphics_compute() {
        let device = VulkanDeviceSummary {
            name: "test".to_string(),
            api_version: "1.0.0".to_string(),
            vendor_id: 0,
            device_id: 0,
            device_type: VulkanDeviceType::DiscreteGpu,
            max_storage_buffer_range: 0,
            timestamp_period: 0.0,
            queue_families: vec![
                VulkanQueueFamilySummary {
                    index: 0,
                    queue_count: 1,
                    graphics: true,
                    compute: true,
                    transfer: true,
                    timestamp_valid_bits: 0,
                },
                VulkanQueueFamilySummary {
                    index: 1,
                    queue_count: 1,
                    graphics: false,
                    compute: true,
                    transfer: true,
                    timestamp_valid_bits: 0,
                },
            ],
        };

        assert_eq!(
            device
                .first_compute_queue_family()
                .map(|family| family.index),
            Some(1)
        );
    }
}
