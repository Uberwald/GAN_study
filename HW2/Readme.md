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
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.skip(x)
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Initial convolution layer
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            ResidualBlock(ndf, ndf * 2, stride=2),
            # state size. ``(ndf*2) x 16 x 16``
            ResidualBlock(ndf * 2, ndf * 4, stride=2),
            # state size. ``(ndf*4) x 8 x 8``
            ResidualBlock(ndf * 4, ndf * 8, stride=2),
            # state size. ``(ndf*8) x 4 x 4``
            ResidualBlock(ndf * 8, ndf * 16, stride=2),
            # state size. ``(ndf*16) x 2 x 2``
            nn.Conv2d(ndf * 16, 1, 2, 1, 0, bias=False),  # Изменен размер ядра на 2
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```




<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%202/Exp2.jpg"
  alt="">
  <figcaption></figcaption>
</figure>   

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%202/Exp2_results.jpg"
  alt="">
  <figcaption></figcaption>
</figure>  

#### Эксперимент 3   
В генератор и дисриминатор были имплементированы spectral norm во все Conv и ConvTranspose слои. Эксперимент не очень удачный

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%203/Exp3.jpg"
  alt="">
  <figcaption></figcaption>
</figure> 

<figure>
  <img
  src="https://github.com/Uberwald/GAN_study/blob/homework_2/HW2/Experiment%203/Exp3_results.jpg"
  alt="">
  <figcaption></figcaption>
</figure>  




# Выводы:   
1) Модель, основанная на свёртках обучается довольно быстро, даже 300 эпох много. В связи с тем, что графики не по эпохам, а по steps, сложно указать точное значение, но примерно 50 эпох должно быть достаточно для обучения модели.
2) Можно поподбирать MSE вручную, немного повысив точность
3) Можно использовать паддинг поменьше - получать изображения меньше, чем 64 на 64. Уменьшение черной может немного повысить точность предсказаний
4) Можно также добавить аугументацию

В целом, можно сказать, что модель рабочая, но нуждается в доработке, так как определение класса 1 всё ещё недостаточно хорошо.



