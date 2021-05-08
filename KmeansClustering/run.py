import sys
sys.path.append('/home/gbazack/Documents/PhD/')
import io
import matplotlib.pyplot as plt
import Simulation
from Simulation.newdataset import load_data
from Simulation.kmeans import _KMeans

def runcode(N):
	inter_rand=[]
	inter_pp=[]
	inter_g=[]
	k_list=[10,20,30,40,50,60,70,80,90,100]
	#-------------------------------------Create a file to store the results
	f=open("results/result.txt","w")
	f2=open("results/clusterValidity.txt","w")	
	f2.write("k\tseparation1\tseparation2\tseparation3\t\t\tcohesion1\tcohesion2\tcohesion3\n")		#writing of the heading on the first line

	for k in k_list:
		Set=load_data()
		
		#run kmeans with random initialization
		krand=_KMeans(k,Set)
		krand.find_centers()
		krand.inter_cluster()
		krand.find_separation()
		#krand.find_cohesion()
		
		#run kmeans++ 
		kpp=_KMeans(k,Set,'++')
		kpp.find_centers()
		kpp.inter_cluster()
		kpp.find_separation()
		#kpp.find_cohesion()

		#run kmeans with geometric initialization 
		kg=_KMeans(k,Set,'geo')
		kg.find_centers()
		kg.inter_cluster()
		kg.find_separation()
		#kg.find_cohesion()

		#Save the results
		f2.write(str(k)+"\t"+str(krand.separation)+"\t"+str(kpp.separation)+"\t"+str(kg.separation)+"\n")
		#inter_rand.append(krand.inter)
		#inter_pp.append(kpp.inter)
		#inter_g.append(kg.inter)
	f.close()

	#plt.hist(inter_rand, bins=20, label="kmeans", facecolor='g')
	#plt.hist(inter_pp, bins=20, label="kmeans++", facecolor='b')
	#plt.hist(inter_g, bins=20, label="geokmeans", facecolor='r')
	
	#plt.axis([10, 100, 0, 5])
	#plt.xlabel('number of clusters',fontsize=18, color='red')
	#plt.ylabel('SSE',fontsize=18, color='red')
	#plt.title("Comparison of SSE",fontsize=18, color='blue')	#Set the title of the graph
		
	#plt.grid(True)
	#plt.legend()
	#plt.savefig("results/resulthisto.png")				#Save the graph as picture 	
	#plt.show()			

				
if __name__=="__main__":
	runcode(2000)

