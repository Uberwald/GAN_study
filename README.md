# ДЗ 4. Обучение Stable diffusion 1.5 методом Dreambooth
## 1. Собрать датасет и обучить Unet
В качестве инференс промта использовался "a photo of sks woman face". Ниже представлены результаты модели CyberRealistic и файн-тюн Dreambooth на фотографиях актрисы Моники Белуччи.  
Промт для генерируемых картинок: "portrait of sks woman face, on the street, lights, midnight, NY, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus,  beautiful eyes, detailed eyes"
<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/Lora_rank4.png"
  alt="Basic CyberRealistic">
  <figcaption>Basic CyberRealistic</figcaption>
</figure>  

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_1.png"
  alt="CyberRealistic + Dreambooth">
  <figcaption>CyberRealistic + Dreambooth</figcaption>
</figure> 

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/Lora_rank4_2.png"
  alt="Basic CyberRealistic">
  <figcaption>Basic CyberRealistic</figcaption>
</figure>  

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_2.png"
  alt="CyberRealistic + Dreambooth">
  <figcaption>CyberRealistic + Dreambooth</figcaption>
</figure>  

## 2. Обучить Lora модель и сравнить с Unet
Дальше была обучена Lora на тот же инстанс промт. Были выбраны параметры rank 4, 32 и 128.


<figure>
  <img
  src="https://developer.mozilla.org/static/img/favicon144.png"
  alt="The beautiful MDN logo.">
  <figcaption>MDN Logo</figcaption>
</figure>


<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/DB_1.png"
  alt="">
</figure>  
