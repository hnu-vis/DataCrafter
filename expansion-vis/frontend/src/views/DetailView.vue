<template>
    <div class="table-header">
        <p>Original</p>
        <p>Generated</p>
    </div>
    <TransitionGroup name="list" class="table-images" tag="div">
        <div class="image-row" v-for="image in selectedImages" :key="image[0].key">
            <div class="original-image">
                <el-popover placement="top" trigger="hover" width="9%" class="popover-content">
                    <template v-slot:reference>
                        <img
                            :src="image[0].key"
                            :alt="image[0].key"
                            :style="
                                dataStore.images.selectedDataContains(image[0])
                                    ? {
                                          borderColor: colorMap[image[0].category],
                                          borderWidth: '2px',
                                          borderStyle: 'solid'
                                      }
                                    : {
                                          opacity: '15%'
                                      }
                            "
                        />
                    </template>
                    <div class="popover-content">
                        <div class="popover-image">
                            <img :src="image[0].key" alt="Original" class="popover-image" />
                            <p><strong>Class</strong>: {{ image[0].category }}</p>
                        </div>
                    </div>
                </el-popover>
            </div>
            <div class="generated-images">
                <el-popover
                    placement="top"
                    trigger="hover"
                    width="18%"
                    v-for="generatedImage in image[1]"
                >
                    <template v-slot:reference>
                        <div class="generated-image">
                            <img
                                :src="generatedImage.key"
                                :style="{
                                    borderColor: generatedColorMap[generatedImage.category],
                                    borderWidth: '2px',
                                    borderStyle: 'dashed'
                                }"
                            />
                        </div>
                    </template>
                    <div class="popover-content">
                        <div class="popover-image">
                            <img :src="generatedImage.originalUrl" alt="Original" />
                            <p><strong>Class</strong>: {{ generatedImage.category }}</p>
                        </div>
                        <div class="popover-image">
                            <img :src="generatedImage.key" alt="Original" />
                            <p><strong>Prompt</strong>: {{ generatedImage.prompt }}</p>
                        </div>
                    </div>
                </el-popover>
                <el-popover
                    placement="top"
                    trigger="hover"
                    width="18%"
                    v-for="similarImage in image[2]"
                >
                    <template v-slot:reference>
                        <div class="similar-image">
                            <img
                                :src="similarImage.key"
                                :style="dataStore.images.isAllSelected() ? '' : 'opacity: 15%'"
                            />
                        </div>
                    </template>
                    <div class="popover-content">
                        <div class="popover-image">
                            <img :src="similarImage.originalUrl" alt="Original" />
                            <p><strong>Class</strong>: {{ similarImage.category }}</p>
                        </div>
                        <div class="popover-image">
                            <img :src="similarImage.key" alt="Original" />

                            <p><strong>Prompt</strong>: {{ similarImage.prompt }}</p>
                        </div>
                    </div>
                </el-popover>
            </div>
        </div>
    </TransitionGroup>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { DataStore, Image } from '../stores/data'
import { config } from '../config/config'

const dataStore = DataStore()
const colorMap = ref(config.petsColorMap)
const generatedColorMap = ref(config.petsGeneratedColorMap)
const selectedImages = ref<[Image, Image[], Image[]][]>([])

watch(
    () => dataStore.images.selectedData(),
    (newSelectedData) => {
        if (dataStore.images.isAllSelected()) {
            const categories = new Set<string>()
            const sampledImages = newSelectedData.filter((image) => {
                if (image.isGenerated) {
                    return false
                }
                if (categories.has(image.category)) {
                    return false
                }
                categories.add(image.category)
                return true
            })
            renderImages(sampledImages)
        } else {
            renderImages(newSelectedData)
        }
    }
)

const renderImages = (images: Image[]) => {
    if (images.length === 0) {
        selectedImages.value = []
        return
    }
    selectedImages.value = dataStore.getImageGroup(images)
}
</script>

<style scoped>
.table-header {
    width: 100%;
    display: grid;
    grid-template-columns: 1fr 4fr;
    column-gap: 32px;
}

.table-header p {
    font-size: 16px;
    font-weight: 500;
}

.table-images {
    width: 100%;
    flex: 1;
    display: flex;
    flex-direction: column;
    row-gap: 8px;
    overflow-y: auto;
}

.image-row {
    width: 100%;
    height: 90px;
    display: grid;
    grid-template-columns: 1fr 4fr;
    column-gap: 32px;
    min-width: 0;
}

.original-image {
    height: 90px;
    display: flex;
}

.generated-images {
    width: 100%;
    height: 90px;
    display: flex;
    overflow-x: auto;
    column-gap: 8px;
}

.generated-images::-webkit-scrollbar {
    height: 10px;
}

.generated-images::-webkit-scrollbar-thumb {
    background-color: #dadada;
    border-radius: 10px;
}

.generated-images::-webkit-scrollbar-track {
    background-color: #f0f0f0;
    border-radius: 10px;
}

.original-image > img,
.generated-image > img,
.similar-image > img {
    width: 80px;
    height: 80px;
    object-fit: cover;
}

.popover-content {
    display: flex;
    flex-direction: row;
    align-items: start;
    justify-content: center;
    column-gap: 8px;
}

.popover-image {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.popover-image img {
    width: 160px;
    height: 160px;
    object-fit: cover;
}

.list-move,
.list-enter-active,
.list-leave-active {
    transition: all 0.8s ease-in-out;
}

.list-enter-from,
.list-leave-to {
    opacity: 0;
    transform: translateX(30px);
}

.list-leave-active {
    position: absolute;
}
</style>
