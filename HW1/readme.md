# ДЗ 2. Имплементация GAN  

Использовался датасет из Kaggle [jessicali9530/celeba-dataset ](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) 
Целью первой части работы являлось сделать генератор лиц, на основе GAN с имплементированным CSPup блоком. 

## 1. Импелементировать CSPup блок

'''   
class CSPupBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ct_pad):
        super(CSPupBlock, self).__init__()

        # Половина каналов для каждого пути
        mid_channels = in_channels // 2

        # Путь A
        self.deconv_a = nn.ConvTranspose2d(mid_channels, out_channels, 4, 2, ct_pad)
        self.bn_a = nn.BatchNorm2d(out_channels)

        # Путь B
        self.conv1_b = nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1)
        self.bn1_b = nn.BatchNorm2d(mid_channels)
        self.relu1_b = nn.ReLU()
        self.deconv_b = nn.ConvTranspose2d(mid_channels, out_channels, 4, 2, ct_pad)
        self.bn2_b = nn.BatchNorm2d(out_channels)

        self.conv3_b = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn3_b = nn.BatchNorm2d(out_channels)
        self.relu3_b = nn.ReLU()
        self.conv3_b_final = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn_final_b = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)

        # Путь A
        out_a = self.bn_a(self.deconv_a(x1))

        # Путь B
        x2 = self.bn1_b(self.conv1_b(x2))
        x2 = self.relu1_b(x2)
        x2 = self.bn2_b(self.deconv_b(x2))
        x2 = self.bn3_b(self.conv3_b(x2))
        x2 = self.relu3_b(x2)
        out_b = self.bn_final_b(self.conv3_b_final(x2))

        out = out_a + out_b
        return out   
'''

## 1.1.2.   
Картинки получились очень похожими, но если сравнивать попиксельно - они разные. Схожесть обуславливается относительно высоким разрешением, если бы их размеры были 16 на 16, то итог был бы визуально более разнообразным.   

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/1.2.Picture/1.png"
  alt="Картинка 1">
  <figcaption>Картинка 1</figcaption>
</figure>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/1.2.Picture/2.png"
  alt="Картинка 2">
  <figcaption>Картинка 2</figcaption>
</figure>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/1.2.Picture/3.png"
  alt="Картинка 3">
  <figcaption>Картинка 3</figcaption>
</figure>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/1.2.Picture/4.png"
  alt="Картинка 4">
  <figcaption>Картинка 4</figcaption>
</figure>

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/1.2.Picture/5.png"
  alt="Картинка 5">
  <figcaption>Картинка 5</figcaption>
</figure>


## 1.2. Детекция протекающих лунок   
Использовался датасет из задания   

#### Эксперимент 1
Цель эксперимента: получить CVAE восстановленные изображения с хорошими значениями метрик.   
Идея эксперимента: для начала взять большое количество эпох, а потом по графику потерь определить их оптимальное количество.   
<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.%20Experiment%201/experiment1_loss.jpg"
  alt="График потерь для 1200 эпох">
  <figcaption>График потерь по steps (у меня не получилось получить по эпохам), 1200 эпох. Можно заметить, что обучение можно было окончить значительно раньше</figcaption>
</figure>   

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.%20Experiment%201/experiment1_metrics.jpg"
  alt="График потерь для 1200 эпох">
  <figcaption>Метрики</figcaption>
</figure>   

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.%20Experiment%201/experiment1_results.jpg"
  alt="">
  <figcaption>Результаты работы CVAE</figcaption>
</figure>


#### Эксперимент 2
Цель эксперимента: получить классификатор, опираясь на полученную модель и среднее значение по выборке proliv.   
Идея эксперимента: обучать модель 300 эпох на датасете train, получить среднее значение по proliv и сделать классификатор для test, давая класс 1 всем картинкам, MSE которых больше среднего по proliv.   

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.2.%20Experiment%202/metrics300.jpg"
  alt="">
  <figcaption>График потерь за 300 эпох</figcaption>
</figure>   

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.2.%20Experiment%202/MSE_train.png"
  alt="">
  <figcaption>График MSE для датасета train. Среднее значение: 0.00040680813253857195</figcaption>
</figure>  

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.2.%20Experiment%202/MSE_proliv.png"
  alt="">
  <figcaption>График MSE для датасета proliv. Среднее значение: 0.0018871622160077095</figcaption>
</figure> 

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.2.%20Experiment%202/MSE_test.png"
  alt="">
  <figcaption>График MSE для датасета test Среднее значение: 0.0013879516627639532</figcaption>
</figure>  

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/main/HW1/2.2.%20Experiment%202/error%20matrix.jpg"
  alt="">
  <figcaption>Матрица ошибок</figcaption>
</figure>   

# Выводы:   
1) Модель, основанная на свёртках обучается довольно быстро, даже 300 эпох много. В связи с тем, что графики не по эпохам, а по steps, сложно указать точное значение, но примерно 50 эпох должно быть достаточно для обучения модели.
2) Можно поподбирать MSE вручную, немного повысив точность
3) Можно использовать паддинг поменьше - получать изображения меньше, чем 64 на 64. Уменьшение черной может немного повысить точность предсказаний
4) Можно также добавить аугументацию

В целом, можно сказать, что модель рабочая, но нуждается в доработке, так как определение класса 1 всё ещё недостаточно хорошо.



