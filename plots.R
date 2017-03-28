library(ggplot2)

bytes = c(1,2,3,4,5,6,7,8,9,10,11)
categs = c('spice','cc','tex')
# Inciso 2 - Cache de Instrucciones
spice = c(0.9519,0.9722,0.9826,0.9885,0.9916,0.9935,0.9946,0.9955,0.9955,0.9889,0.9864)
cc = c(0.8225,0.8982,0.9368,0.9578,0.9696,0.9772,0.982,0.9859,0.9872,0.9885,0.9877)
sc = append(spice,cc)
tex = c(0.9997,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.9999,0.975)
all = append(sc,tex)
df <- data.frame(block_size=rep(bytes, 3), hit_ratio=all, 
                 trace=rep(categs, each=11))

p <- ggplot(data = df, aes(x=block_size, y=hit_ratio)) + geom_line(aes(colour=trace))
p + ggtitle("Instrucciones")



spice = c(0.9701,0.983,0.9886,0.9912,0.9855,0.9814,0.9763,0.9564,0.9415,0.8986,0.7947)
cc = c(0.9339,0.9544,0.9646,0.9695,0.969,0.962,0.9508,0.9267,0.8842,0.8175,0.7062)
tex = c(0.9365,0.9682,0.9841,0.992,0.9959,0.9976,0.9969,0.9919,0.9633,0.91,0.8095)
sc = append(spice,cc)
all = append(sc,tex)
df <- data.frame(block_size=rep(bytes, 3), hit_ratio=all, 
                 trace=rep(categs, each=11))

p <- ggplot(data = df, aes(x=block_size, y=hit_ratio)) + geom_line(aes(colour=trace))
p + ggtitle("Datos")
