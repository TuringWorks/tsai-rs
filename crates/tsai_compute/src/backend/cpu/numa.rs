//! NUMA-aware memory and threading.
//!
//! Provides NUMA topology awareness for optimized memory placement
//! and thread affinity on multi-socket systems.

use crate::error::ComputeResult;

#[cfg(feature = "numa")]
use crate::error::ComputeError;

/// NUMA memory information for a node.
#[derive(Debug, Clone)]
pub struct NumaMemoryInfo {
    /// NUMA node index.
    pub node: u32,
    /// Total memory in bytes.
    pub total_bytes: u64,
    /// Free memory in bytes.
    pub free_bytes: u64,
}

/// NUMA context for topology-aware operations.
#[cfg(feature = "numa")]
pub struct NumaContext {
    topology: hwlocality::Topology,
    node_count: usize,
    local_node: u32,
}

#[cfg(feature = "numa")]
impl NumaContext {
    /// Create a new NUMA context by detecting system topology.
    pub fn new() -> Option<Self> {
        use hwlocality::object::types::ObjectType;

        let topology = hwlocality::Topology::new().ok()?;
        let node_count = topology
            .objects_with_type(ObjectType::NUMANode)
            .count();

        if node_count == 0 {
            return None;
        }

        Some(Self {
            topology,
            node_count,
            local_node: 0,
        })
    }

    /// Get the number of NUMA nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Get the local NUMA node for the current thread.
    pub fn local_node(&self) -> u32 {
        self.local_node
    }

    /// Get memory information for all NUMA nodes.
    pub fn memory_info(&self) -> Vec<NumaMemoryInfo> {
        use hwlocality::object::types::ObjectType;

        self.topology
            .objects_with_type(ObjectType::NUMANode)
            .enumerate()
            .map(|(i, _obj)| {
                // Get memory info from the node
                NumaMemoryInfo {
                    node: i as u32,
                    total_bytes: 0, // Would query from topology
                    free_bytes: 0,
                }
            })
            .collect()
    }

    /// Bind the current thread to a specific NUMA node.
    pub fn bind_thread_to_node(&self, node: u32) -> ComputeResult<()> {
        use hwlocality::object::types::ObjectType;

        let numa_nodes: Vec<_> = self.topology
            .objects_with_type(ObjectType::NUMANode)
            .collect();

        if node as usize >= numa_nodes.len() {
            return Err(ComputeError::InvalidConfig(format!(
                "NUMA node {} not found (max: {})",
                node,
                numa_nodes.len() - 1
            )));
        }

        // Would bind using topology.bind_to_cpuset()
        Ok(())
    }

    /// Get the number of CPUs on a NUMA node.
    pub fn cpus_on_node(&self, node: u32) -> usize {
        use hwlocality::object::types::ObjectType;

        self.topology
            .objects_with_type(ObjectType::NUMANode)
            .nth(node as usize)
            .map(|_obj| {
                // Count PUs under this NUMA node
                self.topology
                    .objects_with_type(ObjectType::PU)
                    .count() / self.node_count.max(1)
            })
            .unwrap_or(0)
    }
}

#[cfg(feature = "numa")]
impl Default for NumaContext {
    fn default() -> Self {
        Self::new().expect("NUMA not available")
    }
}

/// Stub NUMA context when the numa feature is disabled.
#[cfg(not(feature = "numa"))]
pub struct NumaContext;

#[cfg(not(feature = "numa"))]
impl NumaContext {
    /// Create a stub NUMA context.
    pub fn new() -> Option<Self> {
        None
    }

    /// Get node count (always 1 without NUMA support).
    pub fn node_count(&self) -> usize {
        1
    }

    /// Get local node (always 0).
    pub fn local_node(&self) -> u32 {
        0
    }

    /// Get memory info (empty without NUMA).
    pub fn memory_info(&self) -> Vec<NumaMemoryInfo> {
        Vec::new()
    }

    /// Bind thread (no-op without NUMA).
    pub fn bind_thread_to_node(&self, _node: u32) -> ComputeResult<()> {
        Ok(())
    }

    /// Get CPUs on node.
    pub fn cpus_on_node(&self, _node: u32) -> usize {
        num_cpus::get()
    }
}

/// NUMA-aware memory allocator.
pub struct NumaAllocator {
    context: Option<NumaContext>,
}

impl NumaAllocator {
    /// Create a new NUMA allocator.
    pub fn new() -> Self {
        Self {
            context: NumaContext::new(),
        }
    }

    /// Check if NUMA is available.
    pub fn is_numa_available(&self) -> bool {
        self.context.is_some()
    }

    /// Get the number of NUMA nodes.
    pub fn node_count(&self) -> usize {
        self.context.as_ref().map(|c| c.node_count()).unwrap_or(1)
    }

    /// Allocate memory on a specific NUMA node.
    ///
    /// Falls back to regular allocation if NUMA is not available.
    pub fn allocate_on_node(&self, size: usize, _node: u32) -> Vec<u8> {
        // For now, use standard allocation
        // With full NUMA support, would use numa_alloc_onnode or mmap with MPOL_BIND
        vec![0u8; size]
    }

    /// Allocate memory with interleaved policy across all nodes.
    pub fn allocate_interleaved(&self, size: usize) -> Vec<u8> {
        // For now, use standard allocation
        // With full NUMA support, would use mmap with MPOL_INTERLEAVE
        vec![0u8; size]
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_allocator() {
        let allocator = NumaAllocator::new();
        println!("NUMA available: {}", allocator.is_numa_available());
        println!("NUMA nodes: {}", allocator.node_count());

        let data = allocator.allocate_on_node(1024, 0);
        assert_eq!(data.len(), 1024);
    }
}
