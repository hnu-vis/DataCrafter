import './assets/main.css';

import { createApp } from 'vue';
import { createPinia } from 'pinia';

import App from './App.vue';
import 'element-plus/dist/index.css';
import vuetify from './plugins/vuetify';
import ElementPlus from 'element-plus';
const app = createApp(App);


app.use(ElementPlus);
app.use(createPinia());
app.use(vuetify);

app.mount('#app');
