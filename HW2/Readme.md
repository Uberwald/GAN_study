# ДЗ 2. Имплементация GAN  

Использовался датасет из Kaggle [jessicali9530/celeba-dataset ](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) 
Целью первой части работы являлось сделать генератор лиц, на основе GAN с имплементированным CSPup блоком. 

## 1. Импелементировать CSPup блок

```python
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
```


## 2. Имплементировать генератор GAN по заданной архитектурной схеме   

```python
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            CSPupBlock(100, 512, 0), 
            CSPupBlock(512, 256, 1),
            CSPupBlock(256, 128, 1),
            CSPupBlock(128, 64, 1),
            CSPupBlock(64, 3, 1),  
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
```

## 3. Обучение GAN и получение сходимости

#### Эксперимент 1
Цель эксперимента: получить работающий GAN с CSPup блоками по заданной архитектуре 
Идея эксперимента: сделать так, чтобы работало, немного поиграться с гиперпараметрами   

Итог: Получена работающая модель. Для дискриминатора был установлен lr = 0.0001, для генератора lr = 0.0002.


<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%201/Exp1.jpg"
  alt="">
</figure>   


<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%201/Exp1_results.jpg"
  alt="">
  <figcaption>Наблюдается mode collapse</figcaption>
</figure>  


#### Эксперимент 2
Цель эксперимента: улучшить GAN
Идея эксперимента: добавив ResNet в дискриминатор 

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%202/Exp2.jpg"
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



