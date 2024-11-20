import axios from 'axios';
import { config } from '../config/config';

const apiClient = axios.create({
    baseURL: config.baseURL,
    timeout: 10000,
    headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
    },
});

apiClient.interceptors.request.use(
    (request) => {
        const token = localStorage.getItem('token');
        if (token) {
            request.headers.Authorization = `Bearer ${token}`;
        }
        return request;
    },
    (error) => {
        return Promise.reject(error);
    }
);

apiClient.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        if (error.response) {
            console.error('Response error:', error.response);
            switch (error.response.status) {
                case 401:
                    console.error('Unauthorized access - redirecting to login');
                    break;
                case 403:
                    console.error('Forbidden');
                    break;
                case 404:
                    console.error('Not found');
                    break;
                default:
                    console.error('Error:', error.message);
            }
        } else {
            console.error('Network error:', error.message);
        }
        return Promise.reject(error);
    }
);

export default apiClient;
