import { defineStore } from 'pinia'

export const DistributionStore = defineStore({
    id: 'distributionStore',
    state: () => ({
        isBrushing: false,
        isZooming: false,
        showPie: true,
        wordMode: 'default' as 'default' | '☆'
    })
});
