<template>
    <div ref="chartRef" class="metric-lineschart"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, Ref, watch } from 'vue'
import * as echarts from 'echarts'
import eventBus from '../utils/eventBus'
import { DataStore } from '../stores/data'
import { formatToFixed } from '../utils/formatter'

const chartRef = ref(null)
const dataStore = DataStore()

let myChart: echarts.ECharts

const option = {
    tooltip: {
        trigger: 'axis',
        triggerOn: 'mousemove',
        formatter: function (params) {
            let result = ``
            params.forEach((item) => {
                result += `${item.marker}${item.seriesName}: ${formatToFixed(item.value, 3)}<br/>`
            })
            return result
        }
    },
    legend: {
        left: 'center',
        top: 'top',
        orient: 'horizontal',
        textStyle: {
            fontSize: 12
        },
        itemWidth: 50,
        itemHeight: 10,
        selected: {},
        data: ['Distance', 'Diversity', 'Informativeness']
    },
    large: true,
    grid: {
        left: '4%',
        right: '7%',
        bottom: '15%',
        top: '23%'
    },
    xAxis: {
        type: 'category',
        boundaryGap: false,
        data: [1, 2, 3],
        axisLine: {
            show: true,
            onZero: true
        },
        splitLine: {
            show: false
        },
        name: 'Iteration',
        nameLocation: 'end',
        nameTextStyle: {
            fontSize: 13
        },
        axisTick: {
            show: false
        }
    },
    yAxis: {
        axisLine: {
            show: false,
            onZero: false
        },
        splitLine: {
            show: false,
            lineStyle: {
                color: ['#aaa', '#ddd'],
                opacity: 0.25
            }
        },
        type: 'value',
        name: 'Δ Rate',
        min: 0,
        max: 2,
        nameLocation: 'end',
        nameTextStyle: {
            fontSize: 13
        },
        axisLabel: {
            show: false
        }
    },
    dataZoom: [
        {
            type: 'inside',
            yAxisIndex: [0],
            filterMode: 'none'
        }
    ],
    animation: true
}

const currentEpoches: Ref<number[]> = ref([])

async function init() {
    myChart = echarts.init(chartRef.value, null, { renderer: 'svg', useDirtyRect: true })
    myChart.setOption(option)
    await renderMetric()
    myChart.on('click', (params) => {
        const index = params.dataIndex
        dataStore.changeIndex(index)
        highlightIndex(index)
    })
}

const metricData = ref<{ Distance: number[]; Diversity: number[]; Informativeness: number[] }>()

function highlightIndex(index: number) {
    renderChart(metricData.value, index)
}

watch(
    () => dataStore.data.length,
    () => {
        renderMetric()
    }
)

async function renderMetric() {
    const newMetricData = dataStore.data.map((item) => item.metric)
    const epochs = Array.from(newMetricData.keys())

    currentEpoches.value = epochs

    const DistanceData = newMetricData.map((item) => item['Distance'])
    const DiversityData = newMetricData.map((item) => item['Diversity'])
    const InformativenessData = newMetricData.map((item) => item['Informativeness'])

    const data = {
        Distance: DistanceData,
        Diversity: DiversityData,
        Informativeness: InformativenessData
    }

    metricData.value = data

    renderChart(data, dataStore.currentDataIndex)
}

function renderChart(
    data: { Distance: number[]; Diversity: number[]; Informativeness: number[] },
    highlightedIndex: number
) {
    const epochs = Array.from(data.Distance.keys())
    const epochData = epochs.length < 3 ? [1, 2, 3] : epochs.map((item) => item + 1)

    const lineStyle = {
        Distance: 'solid',
        Diversity: 'dashed',
        Informativeness: 'dotted'
    }

    let maxValue = 0
    let minValue = 0

    Object.values(data).forEach((value) => {
        value.map((val) => {
            if (val > 1) {
                if (Math.abs(val - 1) > maxValue) {
                    maxValue = Math.abs(val - 1)
                }
            } else {
                if (Math.abs(val - 1) > minValue) {
                    minValue = Math.abs(val - 1)
                }
            }
        })
    })

    const series = Object.entries(data).map(([key, value]) => ({
        id: key,
        name: key,
        type: 'line',
        data: value.map((val, idx) => ({
            value: val,
            name: `${idx} - ${key}`,
            itemStyle: idx === highlightedIndex ? { color: 'rgb(88, 166, 205)' } : {}
        })),
        smooth: true,
        itemStyle: {
            color: 'rgb(110,110,110,0.5)'
        },
        lineStyle: {
            type: lineStyle[key],
            color: 'rgba(110,110,110,0.5)'
        },
        symbolSize: 8
    }))

    myChart.setOption({
        xAxis: {
            data: epochData
        },
        yAxis: {
            name: 'Δ Rate',
            min: maxValue === minValue && maxValue === 0 ? 0 : 1 - minValue / 0.8,
            max: maxValue === minValue && maxValue === 0 ? 2 : 1 + maxValue / 0.8
        },
        series: series
    })
}

onMounted(async () => {
    await init()
    eventBus.on('onGeneratingData', renderMetric)
})
</script>

<style scoped>
.metric-lineschart {
    width: 100%;
    height: 100%;
    padding: 10px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    background-color: white;
    border-radius: 10px;
}

.edit-menu {
    position: absolute;
    top: 15px;
    right: 16px;
    height: 28px;
    display: flex;
    gap: 32px;
}

.tool-group {
    display: flex;
    align-items: center;
    width: 100%;
    height: 100%;
    color: black;
    gap: 8px;
}

.tool-group > h1 {
    font-size: 16px;
    line-height: 16px;
    padding-right: 6px;
}

.tool-group > button {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.tooltip {
    visibility: hidden;
    color: rgb(88, 166, 205);
    font-size: 14px;
    text-align: center;
    border-radius: 5px;
    padding: 0;
    position: absolute;
    z-index: 1;
    bottom: 80%;
    opacity: 0;
    transition: opacity 0.3s;
}

button:hover .tooltip {
    visibility: visible;
    opacity: 1;
}
</style>
