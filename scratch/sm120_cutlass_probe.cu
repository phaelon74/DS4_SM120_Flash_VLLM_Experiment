#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

using ElementInputA = cutlass::float_e4m3_t;
using ElementInputB = cutlass::float_e2m1_t;
using ElementA = cutlass::mx_float8_t<ElementInputA>;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;

using ElementB = cutlass::mx_float4_t<ElementInputB>;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128;

using ElementC = void;
using LayoutC = cutlass::layout::RowMajor;
using ElementD = cutlass::bfloat16_t;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentC = 1;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ElementCompute = float;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC*, AlignmentC,
    ElementD, LayoutD*, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA*, AlignmentA,
    ElementB, LayoutB*, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

int main() {
    return int(sizeof(typename Gemm::Arguments) == 0);
}
