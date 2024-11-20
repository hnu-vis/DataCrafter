import apiClient from "./apiClient";

export const changeStep = async (dataset: string, step: number) => {
    try {
        const stepData = { step }
        const datasetData = { case: dataset }

        const stepResponse = await apiClient.post(`/changeStep`, stepData)
        const datasetResponse = await apiClient.post(`/changeCase`, datasetData)

        return [stepResponse.data, datasetResponse.data]
    } catch (error) {
        console.error('Error changing step:', error.response ? error.response.data : error.message)
        throw error
    }
}

export const getImagesData = async () => {
    try {
        const response = await apiClient.get('/getImagesData');
        return response.data;
    } catch (error) {
        console.error('Error fetching image data:', error);
        throw error;
    }
}

export const getWordsData = async () =>{
    try {
        const response = await apiClient.get(`/getWordsData`);
        return response.data;
    } catch (error) {
        console.error('Error fetching words data:', error);
        throw error;
    }
}

export const getMetricData = async(key: Object) =>{
    try {
        const response = await apiClient.get(`/getMetricData`,
            {
                params: key,
            });
        return response.data;
    } catch (error) {
        console.error('Error fetching metric data:', error);
        throw error;
    }
}