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
Пси брался 0.1 или 0.01

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

Возможно стоило взять больший пси, хотя в этом случае черты лица Даддарио терялись


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

## 4. Face swap 
Я не успеваю реализовать полный перенос с фото на фото, но суть переноса такова: из обоих изображений вырезается лицо, потом меняется по векторам [4,6,7,8,9,10,11,13] с psi=0.1. Потом снимается маска с полученного векторного изображения, находятся ключевые точки типа бровей, губ, носа, и потом эта маска переносится на оригинальное изображение, с подгоном под ключевые точки, края маски (наверное) надо будет размазать. Хотел сделать функцию чтобы всё сразу делалось, но лучше приступить к следующим лабам.


<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_3/HW3/Pictures/face_swap_1.jpg"
  alt="">
</figure>



# Выводы:   
Работа с изображениями через StyleGAN представляет собой ориентирование в скрытом пространстве обученной модели, чем больше будет получено векторов при обучении, тем мелкие детали мы сможем изменять. В зависимости от ветора, мы можем поменять как фон, так и черты лица, положение частей тел и т.п. Сложность заключается в выявлении векторов, отвечающих за тот или иной элемент, потому что, например, за цвет глаз в разной степени может отвечать целых три вектора, таким образом, мы не получим замены только цвета глаз, по крайней мере в 18-мерном пространстве.




