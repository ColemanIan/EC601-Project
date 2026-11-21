Sprint 2 has focused on the dataset, which has been slower to set up than I originally thought.

I have been following the guide I had Claude produce, but it is not immediately obvious how these multiple datasets should be set up.

The intuition about how to do this is that different sections of the CAGE model train on different data. The sections are trained individually while the rest of the model is either not in use or frozen. Each section is trained on the relevant data, then the weights are frozen and training moves to the next section with different inputs. The model remains end-to-end by having all the modules connected by their compute graph. The end result is a model that is end-to-end and appears simply constructed but actually has been trained in parts on dissimilar data. 

I wonder if training submodules on intrinsically different data that also describes a different scene with possibly mismatched categories aids or diminishes the generalizability of the model. 