# ДЗ 3. Sampling в латентном пространстве StyleGAN

Использовался датасет картинок, собранных вручную из интернета.
   Блокнот слишком много весил, поэтому вот ссылка: https://colab.research.google.com/drive/1KceUTpZGO3Q0c-ExUg3TEciORj1K_LML?usp=sharing   

Почему-то при загрузке в GitHub изображения бледнеют.

## 1. Найти проекции изображений в пространстве StyleGAN

В первом столбце оригинал   
Во втором представлен обычный результат поиска в пространстве.   
В третьем результат использования энкодера e4e_ffhq_encode.pt.   
В четвертом картинки, являющиеся улучшением полученных с помощью энкодера: был произведен поиск вектора, внедерен scheduler.

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/Celebs.jpg"
  alt="">
</figure>  

```python
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from tqdm import tqdm

lpips_loss = Lpips_loss(device)
rec_loss = Rec_loss()

noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
reg_loss = Reg_loss(noise_bufs)

target_tensor = image2tensor_norm(target_pil).to(device).unsqueeze(0)

regularize_noise_weight = 5e5
rec_weight = 0.5
lpips_weight = 1

num_steps = 200
learning_rate = 0.01
initial_latent_vector = nn.Parameter(initial_latent_vector, requires_grad=True)

optimizer = torch.optim.Adam([initial_latent_vector], lr=learning_rate)
scheduler = StepLR(optimizer, step_size=150, gamma=0.01)

generated_tensors = []
for step in tqdm(range(num_steps)):
    synth_tensor = G.synthesis(initial_latent_vector, noise_mode='const')

    lpips_value = lpips_loss(synth_tensor, target_tensor)
    rec_value = rec_loss(synth_tensor, target_tensor)
    reg_value = reg_loss()

    loss = lpips_value*lpips_weight + rec_value*rec_weight + reg_value*regularize_noise_weight

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Обновление learning rate

    generated_tensors.append(synth_tensor)

generated_tensor = G.synthesis(initial_latent_vector, noise_mode='const', force_fp32=True)

# save_image(generated_tensor, path="./projected_image.png")
print(loss.item())
```




## 2. Style transfer

Брались векторы [8,9,12,13,14,15,16,17] из скрытого пространства. Вектор 11 отвечал за цвет глаз, я его сохранил.

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/van_gog.jpg"
  alt="">
</figure> 

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/malevich.jpg"
  alt="">
</figure>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/da_vinchi.jpg"
  alt="">
</figure>


## 3. Expression Transfer

Использовались векторы [2,3,4,5]
Для улыбки только [4]

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/smile.jpg"
  alt="">
</figure>


<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/scary.jpg"
  alt="">
</figure>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/sad.jpg"
  alt="">
</figure>

# Выводы:   
1) Само по себе имплементирование CSPup блока на улучшило результаты DCGAN, а даже ухудшило. Поэтому применение такого инструмента требует тонкой настройки гиперпараметров и применения других инструментов для улучшения результата.
2) Изменение learning rate в соотношении 1:2 в целом улучшило резульат, но недостаточно. Имплементирование ResNet в дискриминатор значительно улучшило результат, хотя и недостаточно.
3) Добавление spectral_norm ко всем Conv и ConvTranspose слоям в генераторе и дискриминаторе дало довольно странный результат.
4) Попытка использования spectral_norm с соотношением learning rate 1:4 (генератор к дискриминатору) [https://sthalles.github.io/advanced_gans/], однако результат ухудшился.

По идее spectral_norm должно помочь улучшить результат, оданоко нужно поиграться с lr и количеством эпох обучения, кажется, что 10 - это много.   
Можно внедрить больше слоев, увеличить размерность скрытого пространства.   
Можно внедрить attention (но я пока не знаю как это работает).  

   P.S. Поставив lrG = 0.0005 и lrD = 0.0001 удалось добиться уже чего-то адеекватного.




