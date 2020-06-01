View(resultNb3final2)
summary(resultNb3final2, digits = 1)

xyplot(Spectral.Efficiency ~ ebno, group = Time.Slot, data = resultNb3final2, type = "l", ylab = "Spectral Efficiency (bps/Hz)", xlab = "Eb/No (dB)", main = "Spectral Efficiency when Nb=3", auto.key = list(columns=3, points=FALSE, lines=TRUE), grid=TRUE)

ggboxplot(resultNb10final, x = "Time.Slot", y = "Spectral.Efficiency", color = "Time.Slot", palette = c("#00AFBB", "#E7B800", "#FC4E07", "#198A51", "#182F23", "#C712BB"))
