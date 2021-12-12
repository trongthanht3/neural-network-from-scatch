# Neural network from scatch  
  
### Project: Neural network from scratch
- Dev by [SCIMTA](https://github.com/SCIMTA) EX-team:
    - [trongthanht3](https://github.com/trongthanht3)  
    - [minhk17khmt](minhk17khmt@gmail.com)  
  
- Thanks to: 
    - Mentor: [Ms. Soan Duong - HVKTQS](soanduong@lqdtu.edu.au)  
    - [Martin Hangan](https://hagan.okstate.edu/)  
    - [Harrison Kinsley & Daniel Kukieła](https://nnfs.io)
  
- Reference:
    - [Neural Network Design - Martin Hangan](https://hagan.okstate.edu/NNDesign.pdf)
    - [Neural Network From Scratch - Harrison Kinsley & Daniel Kukieła](https://nnfs.io)
  
# How to run this???
  
```
  # Clone source code
  git clone https://github.com/trongthanht3/neural-network-from-scatch
  
  cd neural-network-from-scatch  
  
  python main.py --help  
    usage: main.py [-h] [--optimizer OPTIMIZER] [--batch_size BATCH_SIZE]
                   [--epochs EPOCHS] [--save_model SAVE_MODEL]
    
    optional arguments:
      -h, --help            show this help message and exit  
      --optimizer OPTIMIZER  
                            choose optimzer to train  
      --batch_size BATCH_SIZE  
                            batch size  
      --epochs EPOCHS       nums of epoch  
      --save_model SAVE_MODEL  
                            True/False save model  
  
  # Example
  python main.py --optimizer=sgd --batch_size=128 --epochs=25 --save_model=False  
```

# Score
![all](https://raw.githubusercontent.com/trongthanht3/neural-network-from-scatch/main/plot_image/all.png)