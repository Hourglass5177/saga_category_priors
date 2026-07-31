#!/bin/bash


# SAGA 物体移除自动化工作流脚本
# 功能：对指定实例 ID 执行完整的物体移除流程

# # 完整工作流（包含交叉掩码）
# ./run_object_removal.sh --data_path ./data/bear --instance_id 5

# # 自定义搜索窗口
# ./run_object_removal.sh --data_path ./data/bear --instance_id 5 --window_size 50

# # 跳过交叉掩码生成
# ./run_object_removal.sh --data_path ./data/bear --instance_id 5 --skip_cross_mask

# # 跳过原始深度图渲染（如果已存在）
# ./run_object_removal.sh --data_path ./data/bear --instance_id 5 --skip_depth
#  bash ./run_object_removal.sh --data_path ./data/shiyanshi --instance_id 24

'''
bash ./run_object_removal.sh --data_path /home/farsee/dev/3dMass/upload/saga_data/bangongshi --instance_id 11
bash ./run_object_removal.sh --data_path /home/farsee/dev/3dMass/upload/EA25F82F-AAD7-4199-9475-D8D093384234/1776326475789.5288 --instance_id 29
~

'''


set -e  # 遇到错误立即退出

# ========== 颜色定义 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ========== 帮助信息 ==========
usage() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}SAGA 物体移除自动化工作流${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    echo "用法: $0 --data_path <路径> --instance_id <ID> [选项]"
    echo ""
    echo "必需参数:"
    echo "  --data_path      数据根路径"
    echo "  --instance_id    要处理的实例 ID"
    echo ""
    echo "可选参数:"
    echo "  --ply_path       PLY 文件路径（默认自动查找最新）"
    echo "  --outputjson_path output.json 路径（默认 data_path/output.json）"
    echo "  --skip_depth     跳过原始深度图渲染"
    echo "  --skip_cross_mask 跳过后处理交叉掩码生成"
    echo "  --window_size    交叉掩码搜索窗口大小（默认 100）"
    echo "  --help           显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --data_path ./data/bear --instance_id 5"
    echo "  $0 --data_path ./data/bear --instance_id 5 --skip_depth"
    echo "  $0 --data_path ./data/bear --instance_id 5 --window_size 50"
    echo ""
    exit 1
}

# ========== 参数解析 ==========
DATA_PATH=""
INSTANCE_ID=""
PLY_PATH=""
OUTPUTJSON_PATH=""
SKIP_DEPTH=false
SKIP_CROSS_MASK=false
WINDOW_SIZE=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --instance_id)
            INSTANCE_ID="$2"
            shift 2
            ;;
        --ply_path)
            PLY_PATH="$2"
            shift 2
            ;;
        --outputjson_path)
            OUTPUTJSON_PATH="$2"
            shift 2
            ;;
        --skip_depth)
            SKIP_DEPTH=true
            shift
            ;;
        --skip_cross_mask)
            SKIP_CROSS_MASK=true
            shift
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            usage
            ;;
    esac
done

# ========== 参数验证 ==========
if [ -z "$DATA_PATH" ] || [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}错误: 缺少必需参数${NC}"
    usage
fi

# 检查数据路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}错误: 数据路径不存在: $DATA_PATH${NC}"
    exit 1
fi

# ========== 路径设置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPTH_OUTPUT_DIR="${DATA_PATH}/inpaint/depth"
MASK_OUTPUT_DIR="${DATA_PATH}/inpaint/${INSTANCE_ID}/mask"
REMOVED_PLY_DIR="${DATA_PATH}/inpaint/${INSTANCE_ID}"
REMOVED_PLY_PATH="${REMOVED_PLY_DIR}/output.ply"
REMOVED_DEPTH_DIR="${DATA_PATH}/inpaint/${INSTANCE_ID}/depth"
CROSS_MASK_DIR="${DATA_PATH}/inpaint/${INSTANCE_ID}/cross_mask"

# 构建通用参数
COMMON_ARGS="--data_path ${DATA_PATH}"
if [ -n "$PLY_PATH" ]; then
    COMMON_ARGS="${COMMON_ARGS} --ply_path ${PLY_PATH}"
fi
if [ -n "$OUTPUTJSON_PATH" ]; then
    COMMON_ARGS="${COMMON_ARGS} --outputjson_path ${OUTPUTJSON_PATH}"
fi

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}SAGA 物体移除自动化工作流${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "数据路径: ${DATA_PATH}"
echo -e "实例 ID: ${INSTANCE_ID}"
echo -e "搜索窗口: ±${WINDOW_SIZE} 帧"
echo -e "================================================${NC}"
echo ""

# ========== 步骤 1: 检查并渲染原始深度图 ==========
echo -e "${YELLOW}[步骤 1/5] 检查原始深度图...${NC}"

# 检查深度输出目录是否存在且包含 .npy 文件
if [ "$SKIP_DEPTH" = true ]; then
    echo -e "${YELLOW}    跳过原始深度图渲染（用户指定）${NC}"
elif [ -d "$DEPTH_OUTPUT_DIR" ] && [ "$(find "$DEPTH_OUTPUT_DIR" -name '*_depth.npy' | head -1)" ]; then
    echo -e "${GREEN}   原始深度图已存在: ${DEPTH_OUTPUT_DIR}${NC}"
else
    echo -e "${YELLOW}    原始深度图不存在，开始渲染...${NC}"
    echo -e "${BLUE}  执行: python edit_object_removal_saga_1.py ${COMMON_ARGS} --save_depth_npy${NC}"

    python "${SCRIPT_DIR}/edit_object_removal_saga_1.py" \
        ${COMMON_ARGS} \
        --save_depth_npy

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   原始深度图渲染完成${NC}"
    else
        echo -e "${RED}   原始深度图渲染失败${NC}"
        exit 1
    fi
fi
echo ""

# ========== 步骤 2: 渲染实例掩码 ==========
echo -e "${YELLOW}[步骤 2/5] 渲染实例 ${INSTANCE_ID} 的掩码...${NC}"

# 检查掩码是否已存在
if [ -d "$MASK_OUTPUT_DIR" ] && [ "$(find "$MASK_OUTPUT_DIR" -name '*_mask.png' | head -1)" ]; then
    echo -e "${GREEN}   实例掩码已存在: ${MASK_OUTPUT_DIR}${NC}"
else
    echo -e "${YELLOW}  开始渲染掩码...${NC}"
    echo -e "${BLUE}  执行: python edit_object_removal_saga_2.py ${COMMON_ARGS} --instance_id ${INSTANCE_ID}${NC}"

    python "${SCRIPT_DIR}/edit_object_removal_saga_2.py" \
        ${COMMON_ARGS} \
        --instance_id ${INSTANCE_ID}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   实例掩码渲染完成${NC}"
    else
        echo -e "${RED}   实例掩码渲染失败${NC}"
        exit 1
    fi
fi
echo ""

# ========== 步骤 3: 删除高斯点并保存新 PLY ==========
echo -e "${YELLOW}[步骤 3/5] 删除实例 ${INSTANCE_ID} 的高斯点...${NC}"

# 检查 PLY 是否已存在
if [ -f "$REMOVED_PLY_PATH" ]; then
    echo -e "${GREEN}   删除后的 PLY 已存在: ${REMOVED_PLY_PATH}${NC}"
    echo -e "${YELLOW}    如需重新生成，请手动删除: ${REMOVED_PLY_PATH}${NC}"
else
    echo -e "${YELLOW}  开始删除高斯点...${NC}"
    echo -e "${BLUE}  执行: python edit_object_removal_saga_0.py ${COMMON_ARGS} --instance_id ${INSTANCE_ID}${NC}"

    python "${SCRIPT_DIR}/edit_object_removal_saga_0.py" \
        ${COMMON_ARGS} \
        --instance_id ${INSTANCE_ID}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   高斯点删除完成${NC}"
    else
        echo -e "${RED}   高斯点删除失败${NC}"
        exit 1
    fi
fi
echo ""

# ========== 步骤 4: 渲染删除后场景的深度图 ==========
echo -e "${YELLOW}[步骤 4/5] 渲染删除后场景的深度图...${NC}"

# 创建输出目录
mkdir -p "${REMOVED_DEPTH_DIR}"

#  检查是否已经生成过深度图
DEPTH_FILE_COUNT=$(find "$REMOVED_DEPTH_DIR" -name '*_depth.npy' 2>/dev/null | wc -l)

if [ "$DEPTH_FILE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}   移除后深度图已存在 (${DEPTH_FILE_COUNT} 个文件): ${REMOVED_DEPTH_DIR}${NC}"
    echo -e "${YELLOW}    如需重新生成，请手动删除: ${REMOVED_DEPTH_DIR}/*.npy${NC}"
else
    echo -e "${YELLOW}  开始渲染删除后场景的深度图...${NC}"
    echo -e "${BLUE}  执行: python edit_object_removal_saga_1.py ${COMMON_ARGS} --ply_path ${REMOVED_PLY_PATH} --output_dir ${REMOVED_DEPTH_DIR} --save_depth_npy${NC}"

    python "${SCRIPT_DIR}/edit_object_removal_saga_1.py" \
        ${COMMON_ARGS} \
        --ply_path ${REMOVED_PLY_PATH} \
        --output_dir ${REMOVED_DEPTH_DIR} \
        --save_depth_npy

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   删除后场景深度图渲染完成${NC}"
    else
        echo -e "${RED}   删除后场景深度图渲染失败${NC}"
        exit 1
    fi
fi
echo ""

# ========== 步骤 5: 生成交叉掩码 ==========
echo -e "${YELLOW}[步骤 5/5] 生成交叉掩码...${NC}"

if [ "$SKIP_CROSS_MASK" = true ]; then
    echo -e "${YELLOW}    跳过交叉掩码生成（用户指定）${NC}"
elif [ -d "$CROSS_MASK_DIR" ] && [ "$(find "$CROSS_MASK_DIR" -name '*.png' | head -1)" ]; then
    echo -e "${GREEN}   交叉掩码已存在: ${CROSS_MASK_DIR}${NC}"
else
    echo -e "${YELLOW}  开始生成交叉掩码...${NC}"
    echo -e "${BLUE}  执行: python edit_object_removal_saga_3.py --data_path ${DATA_PATH} --instance_id ${INSTANCE_ID} --window_size ${WINDOW_SIZE}${NC}"

    python "${SCRIPT_DIR}/edit_object_removal_saga_3.py" \
        --data_path ${DATA_PATH} \
        --instance_id ${INSTANCE_ID} \
        --window_size ${WINDOW_SIZE}

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}   交叉掩码生成完成${NC}"
    else
        echo -e "${RED}   交叉掩码生成失败${NC}"
        exit 1
    fi
fi
echo ""

# ========== 完成总结 ==========
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN} 自动化工作流完成！${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${BLUE}输出文件汇总:${NC}"
echo -e "  1. 原始深度图:    ${DEPTH_OUTPUT_DIR}/"
echo -e "     └─ *_depth.npy"
echo ""
echo -e "  2. 实例掩码:      ${MASK_OUTPUT_DIR}/"
echo -e "     └─ *_mask.png"
echo ""
echo -e "  3. 删除后模型:    ${REMOVED_PLY_PATH}"
echo ""
echo -e "  4. 删除后深度图:  ${REMOVED_DEPTH_DIR}/"
echo -e "     └─ *_depth.npy"
echo ""
echo -e "  5. 交叉掩码:      ${CROSS_MASK_DIR}/"
echo -e "     └─ *.png"
echo ""
echo -e "${BLUE}================================================${NC}"