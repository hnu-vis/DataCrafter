import { defineStore } from 'pinia'


export const InfoStore = defineStore({
    id: 'infoStore',
    state: () => {
        return {
            currentStep: 3
        };
    },
});