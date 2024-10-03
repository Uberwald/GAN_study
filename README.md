# ДЗ 4. Обучение Stable diffusion 1.5 методом Dreambooth
## 1. Собрать датасет и обучить Unet
В качестве инференс промта использовался "a photo of sks woman face". Ниже представлены результаты модели CyberRealistic и файн-тюн Dreambooth на фотографиях актрисы Моники Белуччи.  
Промт для генерируемых картинок: "portrait of sks woman face, on the street, lights, midnight, NY, standing, 4K, raw, hrd, hd, high quality, realism, sharp focus,  beautiful eyes, detailed eyes"
<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/Lora_rank4.png"
  alt="Basic CyberRealistic">
  <div align="center"><figcaption>Basic CyberRealistic</figcaption></div>
</figure>   

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_1.png"
  alt="CyberRealistic + Dreambooth">
  <div align="center"><figcaption>CyberRealistic + Dreambooth</figcaption></div>
</figure> 

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/Lora_rank4_2.png"
  alt="Basic CyberRealistic">
  <div align="center"><figcaption>Basic CyberRealistic</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_2.png"
  alt="CyberRealistic + Dreambooth">
  <div align="center"><figcaption>CyberRealistic + Dreambooth</figcaption></div>
</figure>



## 2. Обучить Lora модель и сравнить с Unet
Дальше была обучена Lora на тот же инстанс промт. Были выбраны параметры rank 4, 32 и 128.    
Изменения почти не заметны, но обратите внимание на уши.
<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_lora_rank4_2.png"
  alt="Lora rank 4">
  <div align="center"><figcaption>Lora rank 4</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_lora_rank32_2.png"
  alt="Lora rank 32">
  <div align="center"><figcaption>Lora rank 32</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/DB_lora_rank128_2.png"
  alt="Lora rank 128">
  <div align="center"><figcaption>Lora rank 128</figcaption></div>
</figure>  

<br><br>


###2.1. Генераиция в разном окружении
Были выбраны несколько локаций: forest, kitchen, office, school, street, truck.
<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/forest.jpg"
  alt="">
  <div align="center"><figcaption>Forest, only dreambooth</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/forestLora.jpg"
  alt="">
  <div align="center"><figcaption>Forest, db + lora rank 128</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/kitchen.jpg"
  alt="">
  <div align="center"><figcaption>kitchen, only dreambooth</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/kitchenLora.jpg"
  alt="">
  <div align="center"><figcaption>kitchen, db + lora rank 128</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/office.jpg"
  alt="">
  <div align="center"><figcaption>office, only dreambooth</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/officeLora.jpg"
  alt="">
  <div align="center"><figcaption>office, db + lora rank 128</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/school.jpg"
  alt="">
  <div align="center"><figcaption>school, only dreambooth</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/schoolLora.jpg"
  alt="">
  <div align="center"><figcaption>school, db + lora rank 128</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/street.jpg"
  alt="">
  <div align="center"><figcaption>street, only dreambooth</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/streetLora.jpg"
  alt="">
  <div align="center"><figcaption>street, db + lora rank 128</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/truck.jpg"
  alt="">
  <div align="center"><figcaption>truck, only dreambooth</figcaption></div>
</figure>  

<br><br>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_4/results/truckLora.jpg"
  alt="">
  <div align="center"><figcaption>truck, db + lora rank 128</figcaption></div>
</figure>  

<br><br>
