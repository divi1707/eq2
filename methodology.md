# Methodology notes (stage 1)

Reference paper: Zhang and Wang, "A Spatiotemporal Model for Global Earthquake Prediction Based on Convolutional LSTM", IEEE TGRS, vol. 61, 2023.

This file separates three things: what the paper did, what this project will do differently, and what is unclear. Every paper detail below comes from the 12-page PDF you supplied. Where the PDF is silent, that is stated.

## 1. What the paper did

**Data**
- Source: USGS catalogue, downloaded month by month with a crawler, up to May 2021.
- Period used: 1973 to 2021, chosen because record counts change sharply around 1973.
- Kept fields: time, longitude, latitude, magnitude type, magnitude. Depth was dropped for the main dataset.
- Magnitude unification: 34 magnitude types were found. Rare types were removed. The rest were grouped into Mw, Ms, mb, ml, md, me and mfa. Each was converted to energy and then to Mw using energy formulas.
- Cut: events with Mw below 4 were removed (a Mw 5 cut was also tested).

**Maps**
- Each map is a grayscale image of the whole globe in latitude and longitude. Earthquakes are mapped to grey values 100 to 255 with a log function. 0 to 100 is left empty so the model can separate "seismic" from "quiet".
- Maps are rescaled to 0 to 1 before training.
- Final map size: 3840 pixels on a side (about 0.094 by 0.047 degrees per pixel).
- One map covers 15 x 30 = 450 days. The input is N = 9 maps. The target is the next single map.
- Samples slide forward by 1 day.

**Model**
- Sequence-to-sequence encoder and decoder written in PyTorch.
- Encoder: three Conv layers alternating with three ConvLSTM layers. Decoder: three ConvLSTM layers alternating with Deconv layers. LeakyReLU activations. The last encoder state initialises the decoder.
- ConvLSTM gates follow Equations 2 to 6 of the paper, which include peephole terms (W_ci, W_cf, W_co applied to the cell state).
- Adam optimiser. Learning rates 0.0001 to 0.001 were compared and the paper reports little difference. 0.0003 was used for the headline result. Hardware: one RTX 3090.

**Augmentation and loss**
- Random rotation of the globe, applied to a whole sample, to reduce map distortion.
- A weight map built from training-period earthquakes (pixels around each event are marked). A weighted MSE plus MAE loss gives these pixels more weight. Weights balance three pixel groups: blank, weighted area, earthquake pixels. The paper says weights were "adjusted after training" but does not give the values.

**Evaluation**
- Split: first 70% of the time axis for training, last 30% for testing, giving 7879 and 806 samples. No validation set is described.
- Output is treated as a map. A pixel above grey 100 counts as "earthquake predicted".
- Metrics: precision, recall, accuracy, CSI, FAR, R score.
- Before scoring, each earthquake pixel is expanded to about 25 to 31 km around it. This is a spatial tolerance.
- Headline result: precision 64.54%, recall 51.83%, accuracy 99.17%.
- Comparisons: only the published numbers of other studies (Table VI). No simple baseline was run on the same data.
- Depth as an extra channel (dataset M4_8) gave no clear improvement.

**What the paper admits**
- The model cannot predict large events. The largest predicted magnitude in the shown map was below 6 while real events reached Mw 8 or more.
- Extreme class imbalance. About 0.4% of pixels hold events above magnitude 4.

## 2. Points where I cannot copy the paper directly

| Paper choice | Issue for this project | Proposed change |
|---|---|---|
| Global map, 3840 pixels | We study a 5 x 10 degree box. | 0.25 degree grid, 20 x 40 cells. |
| 450-day maps, 9 inputs | Your brief fixes monthly maps and 12 inputs. | Follow the brief. It is a different setting, so paper numbers are not comparable. |
| 1-day sliding stride with 450-day maps | Neighbouring samples share almost all their data. Test samples are far from independent. | Monthly stride. State the effective sample size honestly. |
| Single 70/30 split | No validation set for early stopping or threshold choice. | Chronological 70/15/15 split as in your brief. |
| Energy conversion of all magnitude types to Mw | Energy-based conversion between mb, ml and Mw is crude and not a standard method. | Keep the catalogue's preferred magnitude, report the type mix per decade, and test sensitivity. Decide after seeing the real data. |
| Mw 4 cut | Needs checking for the Himalaya in USGS data. | Download from M 2.5, then study completeness before fixing a threshold. |
| Grey-value map output and a threshold of 100 | Mixes occurrence and magnitude in one number. | Two separate heads: occurrence probability and conditional magnitude. |
| Random globe rotation | Meaningless for a small regional box. | Not used. |
| Tolerance of about 25 to 31 km in scoring | Raises precision and recall. | Report strict cell-level scores. Add tolerance scores separately and label them. |
| No simple baselines | Cannot tell whether the network adds anything. | Per-cell historical rate, persistence and regional rate baselines. |

## 3. Details the paper does not give (as far as the supplied PDF shows)
- Hidden channel counts, kernel sizes, batch size, number of epochs, early stopping rule.
- The exact energy-to-magnitude formulas used.
- The exact loss weights.
- Whether events marked as aftershocks were removed (they were not, as far as the text says).
- The 7 x 13 pixel expansion in one place and "six pixels in latitude and three in longitude" in another are described slightly differently. I read this as the same tolerance but the text is not fully clear.

For these I will choose values myself and mark them as my own choices in the code and README.

## 4. Things to check about the study area before training
- The box 26.5 to 31.5 N and 78 to 88 E contains 20 x 40 = 800 cells at 0.25 degrees.
- The western edge of Uttarakhand reaches slightly west of 78 E, so a small part of the state lies outside the box. Nepal and most of Uttarakhand are inside.
- I have not counted events yet. Whether monthly cell-level targets have enough positive examples must be measured from the real catalogue. If positives are too rare, the fallback is a coarser grid (for example 0.5 degrees) or a longer time step, and I will say so rather than claim the model works.

## 5. What stage 1 delivers
- Config, project layout, USGS downloader with validation, local-CSV fallback, inspection report, unit tests using mock responses.
- No real catalogue has been downloaded in my workspace because the USGS service was blocked from it (HTTP 403 through the sandbox proxy). All tests use clearly labelled synthetic fixtures. No results exist yet.
