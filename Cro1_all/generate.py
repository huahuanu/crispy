# batch_size d_model drop_rate num_layers  lr
for a in [32, 64]:
    for b in [64, 128]:
        for c in [0.1, 0.3]:
            for d in [2, 3]:
                for e in range(1,7):
                    e = e / 10000
                    ss = 'python main.py --batch_size ' + str(a) + ' --d_model ' + str(b) + ' --drop_rate ' + str(
                        c) + ' --num_layers ' + str(d) + ' --lr ' + str(e) + '\n'
                    with open('test.sh', 'a') as f:
                        f.write(ss)
