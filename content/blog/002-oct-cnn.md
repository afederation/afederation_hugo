---
title: "How fast is fast.ai? Building a world-class pathologist in an evening"
date: 2019-05-28
draft: False
type: "Blog"
---

## Background

I'm working through the [fast.ai](http://course.fast.ai) deep learning course this summer, and the instructor Jeremy often brags (in a good way, like a proud dad) about how students in the course can quickly get near-world-class models built with the fast.ai library. I wanted to put this idea to the test and decided to apply approaches *from the first lecture only* to a highly-cited cell paper from last year.

As a biologist, a Cell paper can be the highlight of one's career. Cell publishes the best work from across all fields of biology, and the [Kermany *et. al.*](http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5) paper from last year represented a breakthrough in applying deep learning to ocular pathology. Briefly, they used an imaging technology called retinal optical coherence tomography (OCT) to image healthy and diseased eyes (3 possible conditions) and employed a panel of expert pathologists to classify them. They had multiple pathologists classify a subset of photos so they could get a measure of how accurate the humans were as well.

![oct image](oct.jpg)

Interestingly, they took a similar approach to what we've been working on - transfer learning with a convolutional neural network. Let's see how our simple approach compares.


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *
```

## Load the data

Luckily, the data are already organized and released on the kaggle platform. They have a nice, pip-able API for accessing their data. You need an API token, and all the details are explained well in the [docs](https://github.com/Kaggle/kaggle-api).


```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/kermany2018
```

## Pre-process the data



```python
src = (ImageList.from_folder(path)
       .split_by_folder(train='train', valid='test')
       .label_from_folder())
tfms = get_transforms()
data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))
```

And we can take a look at the images and labels to make sure all looks good.

```python
data.show_batch(rows=3, figsize=(12,9))
```
![png](output_11_0.png)


## Train the model

We're trying to classify the eyes into one of 4 categories.
1. Healthy
2. Choroidal Neovascularization (blood vessel formation in the eye, related to macular degeneration)
3. Diabetic macular edema (fluid in the retina)
4. Drusen (fat deposits in the retina)

We're using the Resnet-34 convolutional neural network architecture pre-trained with the [imagenet](http://www.image-net.org/) dataset.

```python
learn = cnn_learner(data, models.resnet34, metrics=[accuracy, error_rate])
learn.fit_one_cycle(4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.427891</td>
      <td>0.228002</td>
      <td>0.903926</td>
      <td>0.096074</td>
      <td>05:41</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.311100</td>
      <td>0.171017</td>
      <td>0.941116</td>
      <td>0.058884</td>
      <td>05:36</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.271785</td>
      <td>0.107773</td>
      <td>0.966942</td>
      <td>0.033058</td>
      <td>05:38</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.255081</td>
      <td>0.090001</td>
      <td>0.974174</td>
      <td>0.025826</td>
      <td>05:39</td>
    </tr>
  </tbody>
</table>



```python
learn.save('stage-1')
```

97% before fine-tuning? That's not a bad start. How does our confusion matrix look? The biggest difficulty the model is having is between Drussen and CNV, which is the same confusion that was most common in the human pathologists, an encouraging sign.


```python
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()
```

![png](output_20_0.png)


## Stepping it up - resnet50

The final strategy presented in Lecture 1 was increasing the size of the network up to a 50-layer CNN and increasing the image size. I'll try that here and let it go for 4 epochs. Keep in mind that in the Cell paper, they trained for 100 epochs to a 96.6% accuracy.

First, we use the learning rate finder to find the learning rate where we're decreasing loss at the fastest rate, then use that to train the last few layers of the network (8 epochs). Then we'll unfreeze the model, train for a few more epochs to fine-tune the whole model (4 more epochs) and see how we're doing.


```python
data = (src.transform(tfms, size=256)
        .databunch(bs=32).normalize(imagenet_stats))
learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate])
```


```python
learn.lr_find()
learn.recorder.plot()
```

![png](output_23_2.png)


```python
lr = 0.001
learn.fit_one_cycle(4, slice(lr))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.340663</td>
      <td>0.109715</td>
      <td>0.964876</td>
      <td>0.035124</td>
      <td>15:52</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.225786</td>
      <td>0.067803</td>
      <td>0.977273</td>
      <td>0.022727</td>
      <td>15:48</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.195158</td>
      <td>0.072908</td>
      <td>0.979339</td>
      <td>0.020661</td>
      <td>15:49</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.177601</td>
      <td>0.044009</td>
      <td>0.985537</td>
      <td>0.014463</td>
      <td>15:49</td>
    </tr>
  </tbody>
</table>



```python
learn.save('rn50-stage-1')
```

And lastly, unfreeze the model and tune the entire set of parameters.

```python
learn.unfreeze()
learn.fit_one_cycle(4, slice(5e-6, lr/5))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.185058</td>
      <td>0.043757</td>
      <td>0.988636</td>
      <td>0.011364</td>
      <td>20:48</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.130295</td>
      <td>0.024978</td>
      <td>0.992769</td>
      <td>0.007231</td>
      <td>20:50</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.118030</td>
      <td>0.026132</td>
      <td>0.992769</td>
      <td>0.007231</td>
      <td>20:48</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.099916</td>
      <td>0.026108</td>
      <td>0.991736</td>
      <td>0.008264</td>
      <td>20:49</td>
    </tr>
  </tbody>
</table>



```python
learn.save('rn50-stage-2')
```

## Conclusion

We can get ~99% accuracy using the same dataset as Kermany *et. al.* with the fast.ai workflow, even for a seemingly complex problem like retinal pathology. The model in the paper is ~96% accurate and the best human pathologist was in the 99% range, so we're definitely in good company with this result.

My take-aways are twofold:
1. The fast.ai library works exceptionally well out of the box
2. Deep learning is an incredibly fast-moving field

This exercise is not meant to take anything away from the original authors. The task of assembling and annotating the dataset is an impressive feat in itself, and their model is incredibly good at triaging cases with a low false-negative rate, which is a key metric for clinical use.

What this exercise does highlight is one of the barriers to entry for someone trying to get into deep learning coming from another scientific field. Things are moving so quickly, it's intimidating to catch up on the basics knowing that in 18 months a Cell paper can get surpassed by a python library. So for me, it has been really encouraging to take the library and do something useful, while still catching up on the basics as I go.

Moving forward, it has been shown that medical imaging models are highly sensitive to the instrumentation and hospital where the images were generated ([ref](https://arxiv.org/abs/1807.00431)). I'm working with a friend at Baylor Medical Center to gather an additional set of OCT images as a test set to compare the two models with data from an unrelated institution. The results of that experiment will be a good test for both models and I'll be sure to provide an update.
