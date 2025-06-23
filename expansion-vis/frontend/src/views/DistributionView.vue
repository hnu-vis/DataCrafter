<template>
    <div class="distribution-container">
        <div>
            <div class="edit-menu">
                <div class="tool-group">
                    <h1>Edit:</h1>
                    <button @click="addPrompts">
                        <svg :width="iconSize" :height="iconSize" viewBox="0 0 1024 1024">
                            <path
                                d="M512 958.017c-119.648 0-232.129-46.368-316.737-130.56C110.623 743.202 64 631.201 64 512.002c0-119.169 46.624-231.2 131.232-315.425 84.608-84.191 197.089-130.56 316.737-130.56s232.129 46.369 316.704 130.56c84.672 84.225 131.263 196.256 131.263 315.392 0.033 119.2-46.591 231.233-131.263 315.455C744.13 911.616 631.648 958.017 512 958.017z m0-828.034c-102.624 0-199.072 39.744-271.583 111.937C167.937 314.048 128 409.984 128 512s39.903 197.952 112.384 270.047c72.512 72.192 168.96 111.937 271.583 111.937 102.593 0 199.072-39.744 271.584-111.937 72.48-72.16 112.416-168.063 112.384-270.08 0-102.016-39.904-197.919-112.384-270.016C711.07 169.76 614.593 129.983 512 129.983z m224 350.018H544V288c0-17.664-14.334-32-32-32s-32 14.336-32 32v192H288c-17.664 0-32 14.337-32 32.002s14.336 32 32 32h192v192c0 17.696 14.337 32.001 32.002 32.001s32-14.303 32-32V544h192c17.696 0 32.001-14.336 32.001-32.001s-14.306-32-32.002-32z"
                                :fill="iconColor"
                                :stroke="iconColor"
                                :stroke-width="iconStrokeWidth"
                            />
                        </svg>
                        <div class="tooltip">Add</div>
                    </button>
                    <button @click="removePrompts">
                        <svg :width="iconSize + 2" :height="iconSize + 2" viewBox="0 0 1024 1024">
                            <path
                                d="M853.333333 298.666667l-44.8 0c2.133333 6.4 2.133333 14.933333 2.133333 21.333333l0 490.666667c0 46.933333-38.4 85.333333-85.333333 85.333333L298.666667 896c-46.933333 0-85.333333-38.4-85.333333-85.333333L213.333333 320c0-6.4 2.133333-14.933333 2.133333-21.333333L170.666667 298.666667c-12.8 0-21.333333-8.533333-21.333333-21.333333s8.533333-21.333333 21.333333-21.333333l149.333333 0L320 170.666667c0-23.466667 19.2-42.666667 42.666667-42.666667l320 0c23.466667 0 42.666667 19.2 42.666667 42.666667l0 85.333333 128 0c12.8 0 21.333333 8.533333 21.333333 21.333333S866.133333 298.666667 853.333333 298.666667zM682.666667 170.666667 362.666667 170.666667l0 85.333333 320 0L682.666667 170.666667zM256 298.666667l0 512c0 23.466667 19.2 42.666667 42.666667 42.666667l426.666667 0c23.466667 0 42.666667-19.2 42.666667-42.666667L768 298.666667 256 298.666667zM661.333333 725.333333c-12.8 0-21.333333-8.533333-21.333333-21.333333L640 426.666667c0-12.8 8.533333-21.333333 21.333333-21.333333s21.333333 8.533333 21.333333 21.333333l0 277.333333C682.666667 716.8 674.133333 725.333333 661.333333 725.333333zM512 746.666667c-12.8 0-21.333333-8.533333-21.333333-21.333333L490.666667 405.333333c0-12.8 8.533333-21.333333 21.333333-21.333333s21.333333 8.533333 21.333333 21.333333l0 320C533.333333 738.133333 524.8 746.666667 512 746.666667zM362.666667 725.333333c-12.8 0-21.333333-8.533333-21.333333-21.333333L341.333333 426.666667c0-12.8 8.533333-21.333333 21.333333-21.333333 12.8 0 21.333333 8.533333 21.333333 21.333333l0 277.333333C384 716.8 375.466667 725.333333 362.666667 725.333333z"
                                :fill="iconColor"
                                :stroke="iconColor"
                                :stroke-width="iconStrokeWidth"
                            />
                        </svg>
                        <div class="tooltip">Delete</div>
                    </button>
                </div>
                <div class="tool-group">
                    <h1>Select:</h1>
                    <button @click="startBrush">
                        <svg :width="iconSize" :height="iconSize" viewBox="-100 0 1200 1024">
                            <path
                                d="M816.109555 935.833067a34.304731 34.304731 0 0 1-18.181507-5.145709s-21.611981-13.378845-58.318043-34.304732a34.304731 34.304731 0 1 1 32.246447-60.376327c39.107394 20.925886 61.748516 34.304731 62.434611 34.304732a34.304731 34.304731 0 0 1-18.181508 63.463752zM629.14877 850.071239a34.304731 34.304731 0 0 1-11.320561-2.058284 964.992089 964.992089 0 0 0-127.956647-34.304731 34.304731 34.304731 0 0 1 13.378845-68.609462 955.043717 955.043717 0 0 1 137.218925 37.392157 34.304731 34.304731 0 0 1-11.320562 66.551178z m-268.606045-54.201475h-3.77352a1074.424181 1074.424181 0 0 1-140.306351-22.98417 34.304731 34.304731 0 0 1 16.466271-66.551178 1004.099482 1004.099482 0 0 0 131.044073 21.268933 34.304731 34.304731 0 0 1-3.430473 68.609462z m-248.709301-91.593632a34.304731 34.304731 0 0 1-31.217306-20.239792 127.270553 127.270553 0 0 1-12.006656-44.59615c-1.715237-15.437129-5.831804-52.829286-12.006656-102.914194a34.304731 34.304731 0 0 1 30.188164-38.078252 34.304731 34.304731 0 0 1 38.078252 29.845117c5.831804 49.74186 10.291419 87.477065 12.006656 102.914193a66.551179 66.551179 0 0 0 5.488757 22.298075 34.304731 34.304731 0 0 1-17.152366 45.625293 34.304731 34.304731 0 0 1-13.378845 5.14571z m-37.04911-274.43785a34.304731 34.304731 0 0 1-34.304731-29.845116l-16.466271-137.218925a34.304731 34.304731 0 1 1 68.609462-7.890088l16.466271 137.218925a34.304731 34.304731 0 0 1-29.845116 38.421299z m205.828387-152.656054a34.304731 34.304731 0 0 1-22.641122-8.576183l-102.914194-90.221443a34.304731 34.304731 0 0 1 44.939198-50.427955l102.914194 90.564491a34.304731 34.304731 0 0 1-21.611981 60.033279z m-240.133118-121.095701a34.304731 34.304731 0 0 1-34.304731-30.531211C0 61.405469 0 44.596151 0 37.392157A38.078252 38.078252 0 0 1 10.291419 11.663609 36.363015 36.363015 0 0 1 34.304731 0c14.407987 0 20.239791 4.459615 60.376327 37.735204a34.304731 34.304731 0 0 1-20.925886 60.719375c0 6.174852 0 13.378845 2.401331 20.925886a34.304731 34.304731 0 0 1-30.53121 38.078251zM434.297897 411.656774a34.304731 34.304731 0 0 1-22.298075-8.233135l-51.457097-44.596151a34.304731 34.304731 0 0 1 44.939198-52.143191l34.304731 28.815974 39.450441-22.298075a34.304731 34.304731 0 0 1 34.304731 59.690232l-60.376327 34.304731a34.304731 34.304731 0 0 1-18.867602 4.459615zM596.902323 323.836663a34.304731 34.304731 0 0 1-16.123224-64.492895l102.914194-55.573665a34.304731 34.304731 0 0 1 34.304731 60.376327l-102.914193 55.573665a34.304731 34.304731 0 0 1-18.181508 4.116568z m205.828387-111.14733a34.304731 34.304731 0 0 1-30.188163-17.83846 34.304731 34.304731 0 0 1 13.721892-46.311387l102.914194-55.573664a34.304731 34.304731 0 0 1 46.654434 13.721892 34.304731 34.304731 0 0 1-13.721892 46.654434l-102.914194 55.230618a31.9034 31.9034 0 0 1-17.83846 4.116567zM1054.870485 137.218925h-5.831805a34.304731 34.304731 0 0 1-28.129879-37.735205 34.304731 34.304731 0 0 1-44.596151-14.407987 34.304731 34.304731 0 0 1 13.378845-46.654434l60.376327-34.304731a34.304731 34.304731 0 0 1 36.363015 2.401331 34.304731 34.304731 0 0 1 14.06494 34.304731l-11.663609 68.609463a34.304731 34.304731 0 0 1-33.961683 27.786832zM1000.669009 450.078073H994.837205a34.304731 34.304731 0 0 1-28.12988-38.421299L994.837205 253.168916a34.304731 34.304731 0 0 1 39.450441-28.129879A34.304731 34.304731 0 0 1 1063.446667 264.832525l-27.100737 156.772622a34.304731 34.304731 0 0 1-35.676921 28.472926zM960.532474 674.774063h-5.831805a34.304731 34.304731 0 0 1-27.786832-39.793488l11.663609-68.609463a34.304731 34.304731 0 1 1 68.609462 11.663609l-11.663608 68.609462a34.304731 34.304731 0 0 1-34.990826 28.12988z"
                                :fill="distributionStore.isBrushing ? iconHoverColor : iconColor"
                                :stroke="distributionStore.isBrushing ? iconHoverColor : iconColor"
                                :stroke-width="iconStrokeWidth"
                            ></path>
                            <path
                                d="M964.992089 763.623317a59.004138 59.004138 0 0 1-41.851772-17.152366 60.03328 60.03328 0 0 1 41.165677-102.914194 60.03328 60.03328 0 0 1 0 120.06656z m0-85.761828a25.385501 25.385501 0 0 0-17.83846 7.54704 26.75769 26.75769 0 0 0-7.203994 18.181508 24.013312 24.013312 0 0 0 7.890088 18.181508 25.042454 25.042454 0 0 0 17.83846 7.54704 25.728548 25.728548 0 0 0 0-51.457096zM853.158665 1023.996226A63.463753 63.463753 0 0 1 789.008818 960.532474a64.492895 64.492895 0 0 1 64.492894-64.149848 63.8068 63.8068 0 0 1 0 127.6136z m0-93.308868A30.531211 30.531211 0 0 0 823.313549 960.532474a29.845116 29.845116 0 0 0 28.815974 29.159021 28.815974 28.815974 0 0 0 30.188164-28.815974 29.502069 29.502069 0 0 0-28.815975-30.188163z"
                                :fill="distributionStore.isBrushing ? iconHoverColor : iconColor"
                                :stroke="distributionStore.isBrushing ? iconHoverColor : iconColor"
                                :stroke-width="iconStrokeWidth"
                            ></path>
                        </svg>
                        <div class="tooltip">Brush</div>
                    </button>
                    <button @click="clearBrush">
                        <svg :width="iconSize" :height="iconSize" viewBox="0 0 1024 1024">
                            <path
                                d="M320.75 957.19H133.54a68.49 68.49 0 0 1-68.49-68.49V702.8H129v156.55a34.25 34.25 0 0 0 34.25 34.25h157.5zM960 702.8v185.9a68.49 68.49 0 0 1-68.49 68.49H704.3v-63.6h157.53a34.25 34.25 0 0 0 34.25-34.25V702.8z m-255.7-636h187.21A68.49 68.49 0 0 1 960 135.3v185.9h-63.92V164.65a34.25 34.25 0 0 0-34.25-34.25H704.3z m-383.55 0v63.6H163.22A34.25 34.25 0 0 0 129 164.65V321.2H65.05V135.3a68.49 68.49 0 0 1 68.49-68.49z"
                                :fill="iconColor"
                                :stroke="iconColor"
                                :stroke-width="iconStrokeWidth"
                            ></path>
                            <path
                                d="M448.6 66.81h127.85v63.6H448.6zM895.02 439.84h63.92v127.2h-63.92zM448.6 893.59h127.85v63.6H448.6zM64 439.84h63.92v127.2H64zM542.31 512l156.24-155.44a20.28 20.28 0 0 0 0-28.63l-0.85-0.85a20.53 20.53 0 0 0-28.77 0L512.69 482.74 356.45 327.09a20.53 20.53 0 0 0-28.77 0l-0.85 0.85a19.9 19.9 0 0 0 0 28.63L483.06 512 326.83 667.44a20.28 20.28 0 0 0 0 28.63l0.85 0.85a20.53 20.53 0 0 0 28.77 0l156.24-155.44 156.23 155.43a20.53 20.53 0 0 0 28.77 0l0.85-0.85a20.28 20.28 0 0 0 0-28.63z"
                                :fill="iconColor"
                                :stroke="iconColor"
                                :stroke-width="iconStrokeWidth"
                            ></path>
                        </svg>
                        <div class="tooltip">Clear</div>
                    </button>
                    <button @click="resetChart">
                        <svg :width="iconSize" :height="iconSize" viewBox="0 0 1024 1024">
                            <path
                                d="M938.336973 255.26894c-16.685369-6.020494-35.090879 2.752226-40.939358 19.437594l-24.770032 69.493701c-29.070385-65.537376-74.998152-123.162103-133.48295-166.337645-185.947253-137.611288-450.848984-100.112212-590.180413 83.942886C81.534688 350.908785 52.980346 460.653788 68.805644 570.742819c15.825298 110.605073 74.48211 208.481102 164.789518 275.394591 75.686209 55.904586 164.273476 83.082815 252.172686 83.082815 128.494541 0 255.26894-57.624727 338.007727-166.853687 36.639006-48.335965 61.581052-102.348396 74.48211-160.833193 3.78431-17.373425-7.224593-34.402822-24.426004-38.187133-17.201411-3.78431-34.402822 7.052579-38.187133 24.426004-10.836889 49.36805-31.994625 95.123803-62.957164 135.891147-118.173694 156.016798-342.996136 187.839409-500.90509 70.869814-76.546279-56.592642-126.086343-139.33143-139.503444-232.907106-13.417101-93.059634 10.664875-185.775239 67.77356-261.11742C318.05409 144.491853 542.704519 112.497228 700.785486 229.466823c57.280699 42.315471 100.112212 100.972283 123.334117 167.197715l-110.261045-43.003528c-16.513355-6.364522-35.090879 1.720141-41.627415 18.233496-6.536536 16.513355 1.720141 35.090879 18.233496 41.627415l162.38132 63.473207c3.78431 1.548127 7.740635 2.236183 11.69696 2.236183 0.516042 0 1.032085-0.172014 1.548127-0.172014 1.204099 0.172014 2.408198 0.688056 3.612296 0.688056 13.245087 0 25.630102-8.256677 30.274483-21.32975l57.796741-161.693264C963.623047 279.694944 955.022342 261.289434 938.336973 255.26894z"
                                :fill="iconColor"
                                :stroke="iconColor"
                                :stroke-width="iconStrokeWidth"
                            ></path>
                        </svg>
                        <div class="tooltip">Reset</div>
                    </button>
                </div>
            </div>
        </div>
        <div class="distribution-plot" ref="chartRef" id="plot"></div>
    </div>
</template>

<script setup lang="ts">
import { DistributionStore } from '../stores/distribution'
import { ref, Ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'

import {
    DataStore,
    Word,
    Image,
    getOriginalNum,
    getGeneratedNum,
    getFontSize,
    getHeight,
    getWidth
} from '../stores/data'

import { config } from '../config/config'
import { DataBase } from '../stores/dataBase'
import { getLighterColor } from '../utils/color'
import eventBus from '../utils/eventBus'

import { wordTree } from '../utils/tree'

const iconSize = ref(18)
const iconStrokeWidth = ref(10)
const iconColor = ref('rgb(119, 119, 119)')
const iconHoverColor = ref('rgb(88, 166, 205)')

const distributionStore = DistributionStore()
const dataStore = DataStore()

const lastClickedWord: Ref<string> = ref('')
const lastClickedImage: Ref<string> = ref('')

const chartRef = ref()
let myChart: echarts.ECharts
const zoomLevel = ref(1)

const commonOption = {
    xAxis: { type: 'value', show: false, min: -0.01, max: 1.01 },
    yAxis: { type: 'value', show: false, min: -0.01, max: 1.01 },
    tooltip: {
        trigger: 'item',
        borderWidth: 1,
        shadowBlur: 5,
        textStyle: {
            color: 'rgba(0, 0, 0)',
            fontSize: 13
        },
        formatter: (params) => {
            const seriesId: string = params.seriesId
            if (seriesId.startsWith('word')) {
                const [x, y, word, original, generated, isParent, hasWord, realWord] = params.value
                return `
                            <div>
                                <div><strong>${realWord}</strong></div>
                                <div>Original: ${original}</div>
                                <div>Generated: ${generated}</div>
                            </div>
                            `
            } else if (seriesId.endsWith('-highlight') && !seriesId.endsWith('-not-highlight')) {
                const [x, y, url, isGenerated, category, prompt] = params.value
                if (!isGenerated) {
                    return `
                    <style>
                        .popover-image {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            width: 120px;
                        }
                        .popover-image img {
                            width: 100px;
                            height: 100px;
                            object-fit: cover;
                        }
                        .popover-image p {
                            max-width: 100%;
                            word-wrap: break-word;
                            white-space: normal;
                            margin: 0;
                            text-align: center;
                        }
                    </style>
                    <div class="popover-image">
                        <img src="${url}"/>

                        <p><strong>Class</strong>: ${category}</p>
                    </div>
                    `
                } else {
                    return `
                    <style>
                        .popover-image {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            width: 120px;
                        }
                        .popover-image img {
                            width: 100px;
                            height: 100px;
                            object-fit: cover;
                        }
                        .popover-image p {
                            max-width: 100%;
                            word-wrap: break-word;
                            white-space: normal;
                            margin: 0;
                            text-align: center;
                        }
                    </style>
                    <div class="popover-image">
                        <img src="${url}"/>
                        <p><strong>Prompt</strong>: ${prompt}</p>
                    </div>
                    `
                }
            }
        }
    },
    brush: {
        toolbox: [null],
        throttleType: 'debounce',
        throttleDelay: 300,
        xAxisIndex: [0],
        yAxisIndex: [0],
        brushStyle: {
            borderWidth: 1,
            color: 'rgba(150,140,180,0.2)',
            borderColor: 'rgba(120,140,180,0.8)'
        },
        removeOnClick: false,
        outOfBrush: { opacity: 1 }
    },
    dataZoom: [
        {
            type: 'inside',
            xAxisIndex: [0]
        },
        {
            type: 'inside',
            yAxisIndex: [0]
        }
    ],
    grid: {
        show: false,
        left: '0%',
        top: '5%',
        bottom: '2%',
        right: '3%',
        containLabel: true
    },
    animation: true,
    animationDuration: 800,
    animationThreshold: 100000
}

let zoomTimeout: ReturnType<typeof setTimeout>

const dataZoomEvent = (params: any) => {
    distributionStore.isZooming = true
    if (!params.end) {
        myChart.setOption({
            animation: false
        })
    }
    if (zoomTimeout) {
        clearTimeout(zoomTimeout)
    }

    const xZoom = params.end ? params : params.batch[0]
    const yZoom = params.end ? params : params.batch[1] || params.batch[0]
    const zoomedArea = (yZoom.end - yZoom.start) * (xZoom.end - xZoom.start)
    const oldZoomLevel = zoomLevel.value
    const newZoomLevel = zoomedArea / (100 * 100)

    if (Math.abs(oldZoomLevel - newZoomLevel) > 0.001) {
        zoomLevel.value = newZoomLevel
    }

    if ((newZoomLevel > 0.7 && oldZoomLevel < 0.7) || (newZoomLevel < 0.7 && oldZoomLevel > 0.7)) {
        console.log('newZoomLevel, oldZoomLevel', newZoomLevel, oldZoomLevel)
        myChart.setOption({
            animation: true
        })
        renderWords(
            dataStore.words,
            lastClickedWord.value,
            distributionStore.showPie,
            distributionStore.wordMode
        )
    }

    zoomTimeout = setTimeout(() => {
        distributionStore.isZooming = false
        myChart.setOption({
            animation: true
        })
    }, 1000)
}

type SelectedSeries = {
    seriesId: string
    seriesIndex: number
    seriesName: string
    dataIndex: number[]
}[]

const getBrushedData = (params) => {
    const brushed = (params.batch[0].selected as SelectedSeries)
        .filter((item) => !item.seriesId.startsWith('word') && item.dataIndex.length !== 0)
        .map((item) => ({
            seriesId: item.seriesId,
            dataIndex: item.dataIndex
        }))
    const brushedSeriesIds = brushed.map((item) => item.seriesId)
    const brushedData = (myChart.getOption().series as any[])
        .filter((item) => brushedSeriesIds.includes(item.id))
        .map((item) => ({
            id: item.id,
            data: item.data
                .filter((_, index) =>
                    brushed
                        .find((brushedItem) => brushedItem.seriesId === item.id)
                        .dataIndex.includes(index)
                )
                .map((dataItem) => dataItem.value)
        }))
    return brushedData
}

const brushEvent = (params) => {
    if (distributionStore.isBrushing) {
        const brushedData = getBrushedData(params)
        const images: string[] = brushedData.flatMap((item) =>
            item.data.map((dataItem) => dataItem[2])
        )
        const words = dataStore.getCaptions(images)

        dataStore.words.select(words)
        dataStore.images.select(images)

        // console.log(dataStore.words.selectedData())
        // console.log(dataStore.images.selectedData())
        eventBus.emit('selectWords')
        eventBus.emit('selectImages')
    }
}

const clickEvent = (params) => {
    if (distributionStore.isBrushing) {
        return
    }
    if ((params.seriesId as string).startsWith('word')) {
        const word: string = params.value[7]
        // const isParent: boolean = params.value[5]

        if (word !== lastClickedWord.value) {
            lastClickedWord.value = word
            const images = dataStore.getRelatedImages(word)
            const words = dataStore.getRelatedWords(word)

            dataStore.words.select(words)
            dataStore.images.select(images)
        } else {
            lastClickedWord.value = ''
            dataStore.words.selectAll()
            dataStore.images.selectAll()
        }
    } else {
        const image: string = params.value[2]

        if (image !== lastClickedImage.value) {
            lastClickedImage.value = image
            const words = dataStore.getCaptions(image)

            dataStore.words.select(words)
            dataStore.images.select(image)
        } else {
            lastClickedImage.value = ''
            dataStore.words.selectAll()
            dataStore.images.selectAll()
        }
    }
    eventBus.emit('selectWords')
    eventBus.emit('selectImages')
}

const groupImagesByCategory = (images: Image[]): Record<string, Image[]> => {
    return images.reduce((groups: Record<string, Image[]>, image: Image) => {
        const category = image.category

        if (!groups[category]) {
            groups[category] = []
        }
        groups[category].push(image)

        return groups
    }, {})
}

const getHighlightedImagesSeries = (highlightedImages: Image[], isAllSelected: boolean) => {
    const colorMap = config.petsColorMap
    const generatedColorMap = config.petsGeneratedColorMap
    const categories = Object.keys(colorMap)
    const groupedHighlightedImages = groupImagesByCategory(highlightedImages)

    const highlightedSeries = categories.map((category) => ({
        type: 'scatter',
        id: `${category}-highlight`,
        data: groupedHighlightedImages[category]?.map((image) => ({
            value: [image.x, image.y, image.key, image.isGenerated, image.category, image.prompt],
            name: image.key,
            symbol: 'circle',
            itemStyle: {
                color: image.isGenerated ? 'rgb(255, 255, 255)' : colorMap[image.category],
                borderColor: image.isGenerated
                    ? isAllSelected
                        ? generatedColorMap[category]
                        : colorMap[category]
                    : '',
                borderWidth: image.isGenerated ? (isAllSelected ? 1.5 : 3) : 0
            },
            tooltip: {
                borderColor: image.isGenerated ? generatedColorMap[category] : colorMap[category]
            },
            symbolSize: isAllSelected ? (image.isGenerated ? 4 : 3) : image.isGenerated ? 6 : 6
        })),
        zlevel: 8,
        xAxisIndex: 0,
        yAxisIndex: 0
    }))
    return highlightedSeries
}

const getNotHighlightedImagesSeries = (notHighlightedImages: Image[], isAllSelected: boolean) => {
    const colorMap = config.petsColorMap
    const generatedColorMap = config.petsGeneratedColorMap
    const categories = Object.keys(colorMap)
    const groupedNotHighlightedImages = groupImagesByCategory(notHighlightedImages)

    const notHighlightedSeries = categories.map((category) => ({
        type: 'scatter',
        id: `${category}-not-highlight`,
        data: groupedNotHighlightedImages[category]?.map((image) => ({
            value: [image.x, image.y, image.key, image.isGenerated, image.category],
            name: image.key,
            symbol: 'circle',
            itemStyle: {
                color: image.isGenerated
                    ? 'rgb(255, 255, 255)'
                    : getLighterColor(generatedColorMap[category], 0.4),

                borderColor: image.isGenerated
                    ? getLighterColor(generatedColorMap[category], 0.4)
                    : '',
                borderWidth: image.isGenerated ? (isAllSelected ? 0.5 : 1) : 0
            },
            tooltip: {
                borderColor: image.isGenerated ? generatedColorMap[category] : colorMap[category]
            },
            symbolSize: image.isGenerated ? 4 : 3
        })),
        zlevel: 4,
        xAxisIndex: 0,
        yAxisIndex: 0
    }))

    return notHighlightedSeries
}

const renderImages = async (images: DataBase<Image>) => {
    console.log('render images')
    const displayedImages = images.displayedData()
    const selectedImages = images.selectedData()

    const isAllSelected = images.isAllSelected()
    const highlightedImages = selectedImages
    const notHighlightedImages = displayedImages.filter(
        (image) => !highlightedImages.includes(image)
    )

    const highlightedSeries = getHighlightedImagesSeries(highlightedImages, isAllSelected)
    const notHighlightedSeries = getNotHighlightedImagesSeries(notHighlightedImages, isAllSelected)

    myChart &&
        setTimeout(() => {
            myChart.setOption({
                series: highlightedSeries
            })
        }, 300)

    myChart &&
        myChart.setOption({
            series: notHighlightedSeries
        })

    myChart &&
        setTimeout(() => {
            myChart.hideLoading()
        }, 1000)
}

const getRealFontSize = (frequency: number) => {
    if (distributionStore.wordMode == '☆') {
        return 12
    }
    return getFontSize(frequency)
}

const capitalizeFirstLetter = (str: string) =>
    str.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())

const getHighlightedWordsSeries = (
    highlightedWords: Word[],
    showPie: boolean,
    wordMode: string
) => {
    const colorMap = config.petsColorMap
    const categories = Object.keys(colorMap)
    const selectedImages = dataStore.images.selectedData()
    const series = [
        {
            type: 'custom',
            id: 'word-highlight',
            data: highlightedWords.map((word) => {
                const realWord =
                    categories.findIndex(
                        (category) =>
                            capitalizeFirstLetter(category) === capitalizeFirstLetter(word.key)
                    ) !== -1
                        ? capitalizeFirstLetter(word.key)
                        : word.key
                const originalNum = getOriginalNum(word, selectedImages)
                const generatedNum = getGeneratedNum(word, selectedImages)
                return {
                    value: [
                        word.x,
                        word.y,
                        wordMode == 'default' ? realWord : '☆',
                        originalNum,
                        generatedNum,
                        word.isParent,
                        true,
                        realWord
                    ],
                    name: realWord,
                    itemStyle: {
                        color: 'rgba(220, 220, 220)'
                    }
                }
            }),

            renderItem: (params, api) => {
                const [x, y]: [number, number] = api.coord([api.value(0), api.value(1)])
                const word: string = api.value(2)
                const originalNum: number = api.value(3)
                const generatedNum: number = api.value(4)
                const frequency = originalNum + generatedNum
                const fontSize = getRealFontSize(frequency)
                const radius = (fontSize / 2) * 0.8
                const opacity = 1
                const z = 10

                const pieChart = showPie
                    ? [
                          {
                              type: 'sector',
                              x: x - radius,
                              y: y,
                              z2: z,
                              shape: {
                                  cx: 0,
                                  cy: 0,
                                  r: radius,
                                  r0: 0,
                                  startAngle: 0,
                                  endAngle: Math.PI * 2 * (originalNum / frequency)
                              },
                              style: {
                                  fill: getLighterColor('rgba(120, 120, 120)', 0.7),
                                  // fill: 'rgba(120, 120, 120)',
                                  opacity: opacity
                              },
                              transition: 'style',
                              enterFrom: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              leaveTo: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              updateAnimation: {
                                  duration: 800
                              },
                              leaveAnimation: {
                                  duration: 800
                              }
                          },
                          {
                              type: 'sector',
                              x: x - radius,
                              y: y,
                              z2: z,
                              shape: {
                                  cx: 0,
                                  cy: 0,
                                  r: radius,
                                  r0: 0,
                                  startAngle: Math.PI * 2 * (originalNum / frequency),
                                  endAngle: Math.PI * 2
                              },
                              style: {
                                  fill: getLighterColor('rgba(200, 200, 200)', 0.7),
                                  // fill: 'rgba(200, 200, 200)',
                                  opacity: opacity
                              },
                              transition: 'style',
                              enterFrom: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              leaveTo: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              updateAnimation: {
                                  duration: 800
                              },
                              leaveAnimation: {
                                  duration: 800
                              }
                          }
                      ]
                    : []

                const textChart = {
                    type: 'text',
                    x: x,
                    y: y,
                    z2: z,
                    style: {
                        text: word,
                        fill: 'black',
                        font: api.font({
                            fontSize: fontSize,
                            fontFamily: 'sans-serif',
                            fontStyle: 'normal',
                            fontWeight: wordMode == 'default' ? 'normal' : 'bold'
                        }),
                        textAlign: 'left',
                        textVerticalAlign: 'middle',
                        backgroundColor:
                            word === lastClickedWord.value
                                ? 'rgba(180, 180, 180, 0.6)'
                                : 'rgba(255, 255, 255, 0.6)',
                        // : '',
                        opacity: opacity,
                        borderRadius: 5
                    },
                    transition: 'style',
                    enterFrom: {
                        style: {
                            opacity: 0
                        }
                    },
                    leaveTo: {
                        style: {
                            opacity: 0
                        }
                    },
                    updateAnimation: {
                        duration: 800
                    },
                    leaveAnimation: {
                        duration: 800
                    }
                }

                return {
                    type: 'group',
                    children: [...pieChart, textChart]
                }
            },
            zlevel: 10
        }
    ]
    return series
}

const getNotHighlightedWordsSeries = (
    notHighlightedWords: Word[],
    showPie: boolean,
    wordMode: string
) => {
    const colorMap = config.petsColorMap
    const categories = Object.keys(colorMap)

    const series = [
        {
            type: 'custom',
            id: 'word-not-highlight',
            data: notHighlightedWords.map((word) => {
                const realWord =
                    categories.findIndex(
                        (category) =>
                            capitalizeFirstLetter(category) === capitalizeFirstLetter(word.key)
                    ) !== -1
                        ? capitalizeFirstLetter(word.key)
                        : word.key
                const originalNum = word.originalNum
                const generatedNum = word.generatedNum
                return {
                    value: [
                        word.x,
                        word.y,
                        wordMode == 'default' ? realWord : '☆',
                        originalNum,
                        generatedNum,
                        word.isParent,
                        false,
                        realWord
                    ],
                    name: realWord,
                    itemStyle: {
                        color: 'rgba(220, 220, 220)'
                    }
                }
            }),

            renderItem: (params, api) => {
                const [x, y]: [number, number] = api.coord([api.value(0), api.value(1)])
                const word: string = api.value(2)
                const originalNum: number = api.value(3)
                const generatedNum: number = api.value(4)
                const frequency = originalNum + generatedNum
                const fontSize = getRealFontSize(frequency)
                const radius = (fontSize / 2) * 0.8
                const opacity = 0.2
                const z = 1

                const pieChart = showPie
                    ? [
                          {
                              type: 'sector',
                              x: x - radius,
                              y: y,
                              z2: z,
                              shape: {
                                  cx: 0,
                                  cy: 0,
                                  r: radius,
                                  r0: 0,
                                  startAngle: 0,
                                  endAngle: Math.PI * 2 * (originalNum / frequency)
                              },
                              style: {
                                  fill: getLighterColor('rgba(120, 120, 120)', 0.7),
                                  opacity: opacity
                              },
                              transition: 'style',
                              enterFrom: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              leaveTo: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              updateAnimation: {
                                  duration: 800
                              },
                              leaveAnimation: {
                                  duration: 800
                              }
                          },
                          {
                              type: 'sector',
                              x: x - radius,
                              y: y,
                              z2: z,
                              shape: {
                                  cx: 0,
                                  cy: 0,
                                  r: radius,
                                  r0: 0,
                                  startAngle: Math.PI * 2 * (originalNum / frequency),
                                  endAngle: Math.PI * 2
                              },
                              style: {
                                  fill: getLighterColor('rgba(200, 200, 200)', 0.7),
                                  opacity: opacity
                              },
                              transition: 'style',
                              enterFrom: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              leaveTo: {
                                  style: {
                                      opacity: 0
                                  }
                              },
                              updateAnimation: {
                                  duration: 800
                              },
                              leaveAnimation: {
                                  duration: 800
                              }
                          }
                      ]
                    : []

                const textChart = {
                    type: 'text',
                    x: x,
                    y: y,
                    z2: z,
                    style: {
                        text: word,
                        fill: 'black',
                        font: api.font({
                            fontSize: fontSize,
                            fontFamily: 'sans-serif',
                            fontStyle: 'normal',
                            fontWeight: wordMode == 'default' ? 'normal' : 'bold'
                        }),
                        textAlign: 'left',
                        textVerticalAlign: 'middle',
                        backgroundColor: 'rgba(255, 255, 255, 0.6)',
                        opacity: opacity,
                        borderRadius: 5
                    },
                    transition: 'style',
                    enterFrom: {
                        style: {
                            opacity: 0
                        }
                    },
                    leaveTo: {
                        style: {
                            opacity: 0
                        }
                    },
                    updateAnimation: {
                        duration: 800
                    },
                    leaveAnimation: {
                        duration: 800
                    }
                }

                return {
                    type: 'group',
                    children: [...pieChart, textChart]
                }
            },
            zlevel: 10
        }
    ]

    return series
}

type WordLabel = {
    word: string
    x: number
    y: number
    height: number
    width: number
}

const isOverlap = (label1: WordLabel, label2: WordLabel) => {
    const paddingRatio = 1.05
    if (Math.abs(label1.x - label2.x) > (paddingRatio * (label1.width + label2.width)) / 2) {
        return false
    }
    if (Math.abs(label1.y - label2.y) > (paddingRatio * (label1.height + label2.height)) / 2) {
        return false
    }
    return true
}

const wordList = Object.entries(wordTree).flatMap(([parent, children]) => [parent, ...children])

type Cache = Map<string, Word[]>

let cache: Cache = new Map()

const memoize = <T extends (...args: any[]) => any>(func: T): T => {
    return ((...args: Parameters<T>) => {
        const key = JSON.stringify(args)
        if (cache.has(key)) {
            console.log('use cache')
            return cache.get(key)!
        }
        const result = func(...args)
        cache.set(key, result)
        return result
    }) as T
}

const filterWords = (words: Word[], zoomLevel: number): Word[] => {
    let filteredWords: Word[]
    if (zoomLevel <= 0.7) {
        filteredWords = words.filter((word) => {
            if (!wordList.includes(word.key)) {
                return false
            }
            if (word.isParent) {
                return false
            }
            return true
        })
    } else {
        const defaultWords = ['kitten', 'box', 'Bengal', 'field', 'toy', 'tongue']
        filteredWords = words.filter((word) => {
            if (!wordList.includes(word.key)) {
                return false
            }
            if (word.isParent) {
                return false
            }
            if (!dataStore.words.isAllSelected() && dataStore.words.selectedDataContains(word)) {
                return true
            }
            if (dataStore.sortedWords.findIndex((item) => item.key === word.key) <= 10) {
                return true
            }
            if (word.generatedNum + word.originalNum >= 100) {
                return true
            }
            if (defaultWords.includes(word.key)) {
                return true
            }
            return false
        })
        const filteredWordTree = Object.entries(wordTree)
            .map(([parent, children]) => [
                parent,
                children.filter(
                    (item) => filteredWords.findIndex((word) => word.key === item) === -1
                )
            ])
            .sort((a, b) => a[1].length - b[1].length)
        const labels = words.map((word) => {
            const fontSize = getRealFontSize(word.generatedNum + word.originalNum)
            const height = getHeight(word.key, fontSize)
            const width = getWidth(word.key, fontSize)
            return {
                word: word.key,
                x: word.x,
                y: word.y,
                height: height,
                width: width
            }
        })
        filteredWordTree.forEach(([parent, children]: [string, string[]]) => {
            const filteredLabels = labels.filter(
                (label) => filteredWords.findIndex((item) => item.key === label.word) !== -1
            )
            for (let childIndex in children) {
                const child = children[childIndex]
                const childLabel = labels.find((item) => item.word === child)
                if (!childLabel) {
                    continue
                }
                for (let labelIndex in filteredLabels) {
                    const label = filteredLabels[labelIndex]
                    if (isOverlap(label, childLabel)) {
                        filteredWords.push(words.find((word) => word.key === parent))
                        return
                    }
                }
            }
            filteredWords.push(
                ...words.filter((word) => children.findIndex((item) => item === word.key) !== -1)
            )
        })
    }
    return filteredWords
}

const cachedFilterWords = memoize(filterWords)

const renderWords = async (
    words: DataBase<Word>,
    clickedWord: string,
    showPie: boolean = true,
    wordMode: 'default' | '☆' = 'default'
) => {
    console.log('render words')
    const displayedWords = cachedFilterWords(words.displayedData(), zoomLevel.value)
    const selectedWords = cachedFilterWords(words.selectedData(), zoomLevel.value)

    const highlightedWords = selectedWords
    const notHighlightedWords = displayedWords.filter((word) => !highlightedWords.includes(word))

    const highlightedSeries = getHighlightedWordsSeries(highlightedWords, showPie, wordMode)
    const notHighlightedSeries = getNotHighlightedWordsSeries(
        notHighlightedWords,
        showPie,
        wordMode
    )

    myChart &&
        setTimeout(() => {
            myChart?.setOption({
                series: highlightedSeries
            })
        }, 300)

    myChart &&
        myChart?.setOption({
            series: notHighlightedSeries
        })

    myChart &&
        setTimeout(() => {
            myChart?.hideLoading()
        }, 1000)
}

const addPrompts = () => {
    eventBus.emit('addPrompts', {
        selectedImages: dataStore.images.selectedData(),
        selectedWords: dataStore.words.selectedData(),
        clickedWord: lastClickedWord.value
    })
}

const removePrompts = () => {
    dataStore.images.hide(dataStore.images.selectedData())
    dataStore.words.hide(lastClickedWord.value)

    eventBus.emit('removePrompts', {
        selectedImages: dataStore.images.selectedData(),
        selectedWords: dataStore.words.selectedData(),
        clickedWord: lastClickedWord.value
    })

    eventBus.emit('displayImages')
    eventBus.emit('displayWords')
}

const removeClick = () => {
    lastClickedWord.value = ''
    lastClickedImage.value = ''
}

const clearSelection = () => {
    dataStore.words.selectAll()
    dataStore.images.selectAll()
    eventBus.emit('clearSelection')
}

const clearBrush = () => {
    myChart.dispatchAction({
        type: 'takeGlobalCursor'
    })

    myChart.dispatchAction({
        type: 'brush',
        areas: []
    })

    distributionStore.isBrushing = false
    removeClick()
    clearSelection()
    eventBus.emit('selectImages')
    eventBus.emit('selectWords')
}

const resetZoom = () => {
    myChart.dispatchAction({
        type: 'dataZoom',
        start: 0,
        end: 100
    })
}

const resetChart = () => {
    setTimeout(() => {
        resetZoom()
    }, 500)
    clearBrush()

    eventBus.emit('selectImages')
    eventBus.emit('selectWords')
}

const startBrush = () => {
    if (myChart == null) {
        return
    }
    if (!distributionStore.isBrushing) {
        myChart?.dispatchAction({
            type: 'takeGlobalCursor',
            key: 'brush',
            brushOption: {
                brushType: 'polygon',
                brushMode: 'single'
            }
        })
        distributionStore.isBrushing = true
    } else {
        myChart?.dispatchAction({
            type: 'takeGlobalCursor'
        })
        distributionStore.isBrushing = false
    }
}

const isDatasetChanged = ref(false)

const handleChangeDataset = () => {
    isDatasetChanged.value = true
}

const handleGenerateDataStart = () => {
    if (isDatasetChanged.value) {
        isDatasetChanged.value = false
        myChart.clear()
        myChart.setOption(commonOption)
        myChart.showLoading('default', { zlevel: 10000 })
    }
}

const handleGenerateDataComplete = () => {
    resetChart()
}

const handleSelectImages = () => {
    myChart && renderImages(dataStore.images)
}

const handleSelectWords = () => {
    myChart &&
        renderWords(
            dataStore.words,
            lastClickedWord.value,
            distributionStore.showPie,
            distributionStore.wordMode
        )
}

const initChart = () => {
    myChart = echarts.init(chartRef.value, null, { renderer: 'svg', useDirtyRect: true })
    myChart.setOption(commonOption)

    resetChart()

    myChart.on('dataZoom', dataZoomEvent)
    myChart.on('brushselected', brushEvent)
    myChart.on('click', clickEvent)
}

onMounted(() => {
    initChart()
    eventBus.on('changeDataset', handleChangeDataset)
    eventBus.on('generateDataStart', handleGenerateDataStart)
    eventBus.on('generateDataComplete', handleGenerateDataComplete)
    eventBus.on('displayImages', handleSelectImages)
    eventBus.on('selectImages', handleSelectImages)
    eventBus.on('displayWords', handleSelectWords)
    eventBus.on('selectWords', handleSelectWords)
})

onUnmounted(() => {
    eventBus.off('changeDataset')
    eventBus.off('generateDataStart')
    eventBus.off('generateDataComplete')
    eventBus.off('displayImages')
    eventBus.off('selectImages')
    eventBus.off('displayWords')
    eventBus.off('selectWords')
})
</script>

<style scoped>
.edit-menu {
    position: absolute;
    display: flex;
    top: 15px;
    right: 16px;
    height: 28px;
    gap: 32px;
}

.tool-group {
    display: flex;
    align-items: center;
    width: 100%;
    height: 100%;
    gap: 8px;
    color: black;
}

.tool-group > h1 {
    font-size: 16px;
    line-height: 20px;
}

.tool-group > button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
}

.tooltip {
    position: absolute;
    bottom: 80%;
    z-index: 1;
    padding: 0;
    font-size: 14px;
    color: #58a6cd;
    text-align: center;
    border-radius: 5px;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s;
}

button:hover .tooltip {
    visibility: visible;
    opacity: 1;
}

.distribution-container {
    width: 100%;
    height: 100%;
    background-color: white;
    border: 1px solid rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}

.distribution-plot {
    width: 100%;
    height: 100%;
    transition: opacity 0.5s ease;
}

.upload-button {
    position: absolute;
    top: 15px;
    left: 150px;
}

img .generated-image {
    width: 100px;
    height: 100px;
    object-fit: cover;
}
</style>
