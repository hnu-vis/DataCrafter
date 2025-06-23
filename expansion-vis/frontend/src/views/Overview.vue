<template>
    <div class="grid-2-cols">
        <Container class="info" title-name="Info">
            <InfoView></InfoView>
        </Container>
        <div class="grid-2-2">
            <Container class="metric" title-name="Metric">
                <MetricView></MetricView>
            </Container>
            <Container class="distribution" title-name="Multi-modal Distribution">
                <DistributionView></DistributionView>
            </Container>
        </div>
        <div class="grid-3-2">
            <Container class="prompt" title-name="Prompt">
                <PromptView></PromptView>
            </Container>
            <Container class="detail" title-name="Detail">
                <DetailView></DetailView>
            </Container>
        </div>
    </div>
</template>

<script setup lang="ts">
import Container from '../components/Container.vue'
import MetricView from './MetricView.vue'
import DistributionView from './DistributionView.vue'
import PromptView from './PromptView.vue'
import DetailView from './DetailView.vue'
import InfoView from './InfoView.vue'
</script>

<style scoped>
.grid-2-cols {
    display: grid;
    grid-template-columns: 5fr 16fr 7fr;
    grid-gap: 8px;
    padding: 8px;

    height: 100vh;
    width: 100vw;

    min-height: 0;
    min-width: 0;
}

.grid-2-2 {
    display: grid;
    grid-template-rows: 1fr 3fr;
    grid-gap: 8px;

    height: 100%;
    width: 100%;

    min-height: 0;
    min-width: 0;
}

.grid-3-2 {
    display: grid;
    grid-template-rows: 1fr 1fr;
    grid-gap: 8px;

    height: 100%;
    width: 100%;

    min-height: 0;
    min-width: 0;
}
</style>
