Mission
---
One sentence, using the template from class. Do not call it final — it is your best guess until your first test says otherwise.

    Improve the regularization of MoGe
Target user
---
One specific person. If your product has several users, name the primary one and build for them first.



User stories
---
Your top 5, with acceptance criteria. Put them on your GitHub board as issues — the board is the plan; the document just explains it.

Feasibility
---
Show me, don’t tell me.  “We will use dataset X” is a wish. Downloaded, loaded by a script, committed to the repo — that is feasibility. Same for API keys and hardware: prove access this week.

Tooling
---
Languages, frameworks, models, and why — one line each. Full setup: next slide.

Demo
---
Try to put it in one sentence.
“At the end of two weeks we will show ____ working end to end.” One sentence. If you cannot write it, your goals are too vague.

Your riskiest assumption (and test)
---
List 3–5 assumptions your project stands on. Pick the one that kills the project if wrong. Test that one first, with the cheapest test you can find. Write your pivot line now: “We change direction if …” — decide it while you are still honest.

Evaluation with a baseline
---
Name your metric AND what you compare against. “Our system finds relevant papers” is marketing. “Beats keyword alerts on recall, on a 50-paper labeled set” is research. No baseline = no claim. If you cannot name a baseline, you have not defined the task yet — that is a finding, write it down.

Related work, as a team.  
---
Each of you read papers for Project 
1. Merge them: 8–10 papers, one line each — what it does, and what it leaves open for you. That open part is your project.
10. Three lines on harm.  Who gets hurt if your system is wrong? What is the worst realistic misuse? Three lines now. We will go deeper in Project 3.

---

TO DO AFTER SPRINT PLANNING
---
- List 2–3 BU professors  whose work touches your project. Use the Research Areas handout as a start — then check their recent papers or lab page yourself. A name without a reason does not count.
- For each one, write:  why they are relevant (cite one paper or project of theirs), and one specific question you would ask them.
- Make the question real.  “Can you help us?” has no answer. “For a lab-alert tool, would you weight recall over precision?” gets a reply. If you cannot write the specific question, you do not understand your project yet.
- If you email them  (optional, encouraged): read their latest abstract first, keep it under 150 words, and ask for nothing that costs more than 15 minutes. Professors answer good questions. They delete vague ones.



Writing
---

To briefly summarize the work of MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement, the model works as follows:
- Take an image
- Input image to a Vision Transformer Encoder (ViT Encoder or ViT) 
- - outputs affine pointmap which is scaled to a scaled and shifted point map, 
- additionally processing the image tokens, N images x D dimensions, on a convolutional encoder path 
- while the DINOv2 image classification token, one tensor of D dimensions, is processed on an Multi-Layer Perceptron (MLP) path, 
- to be combined to a metric scale point map and resulting depth.
-//MoGe-3 part
- Voxelize the point map to a 3D structure that is naturally sparse as the problem of reconstruction of a 3D space from a 2D space is ill posed
- Use a sparse 3D U-net where ViT Encoder features (outputs) are concatenated with the bottleneck features and then decoded to produce a directly refined point map. 
- This point map is then fed through an algorithmic Self-Guided Sparse 3D Refiner (SSR) which performs K iterations of regularization, producing better and better point maps, ideally
- output final pointmap


My Idea
---
- Produce an initial point cloud from a single image (ViT Encoder)
- Take advantage of known Field of View (FOV) charachteristics of a camera to project frustums over objects for classification (also input the original image)
- use the object classifications and bounding boxes as a prior for object-wise regularization of the MoGe-3 SSR method
- also use a better concatenation method in the Sparse 3D U-Net
\#READ Frustum PointNets

Extra: Use highly accurate 3D model synthetic textures of real objects, eg. a person with hundreds to thousands of hairs on their head, as an extreme example, and fine-tune the ViT to predict the pointmap and pointmap density (local energy), object masks, and object classifications -> use this as a better prior for object-wise regularization

