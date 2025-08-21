// sw.js
const CACHE = 'audio-precache-v1';
// 把你页面里的音频清单列出来（相对路径即可）
const TO_CACHE = [
  'assets/audio/1.5b_4p_90min.mp3',
  'assets/audio/gpt5_topic_12min.mp3',
  'assets/audio/chinese_demo.mp3',
  'assets/audio/2p_see_u_again.mp3',
];

self.addEventListener('install', (event) => {
  // 安装阶段就开始“下载全部音频”到 Cache Storage
  event.waitUntil(
    caches.open(CACHE).then(cache => cache.addAll(TO_CACHE))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  // 如需清理旧版本：在这里删除旧 cache（略）
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // 只拦截我们的音频请求：命中就优先走缓存（无则网络并回灌缓存）
  if (TO_CACHE.some(p => url.pathname.endsWith(p))) {
    event.respondWith(
      caches.match(event.request).then(resp => resp || fetch(event.request).then(net => {
        const clone = net.clone();
        caches.open(CACHE).then(c => c.put(event.request, clone));
        return net;
      }))
    );
  }
});
