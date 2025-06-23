<template>
    <div class="templates">
        <div class="title">
            <h1>{{ titleName }}</h1>
            <hr />
        </div>
        <slot></slot>
    </div>
</template>

<script setup lang="ts">
defineProps({
    titleName: String
})
</script>

<style scoped>
.templates {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    min-height: 0;
    min-width: 0;
    padding: 16px;
    gap: 8px;
    overflow: hidden;

    background: #ffffff;
    border-radius: 10px;
    box-shadow: 4px 4px 10px #0000001a;

    will-change: filter;
    transition: filter 300ms;
}


.templates:hover {
    filter: drop-shadow(0 0 0.5em #9e9e9eaa);
}

.title {
    width: 100%;
}

.title h1 {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    text-align: left;
    font-size: 20px;
}

.content {
    flex: 1;
}
</style>
