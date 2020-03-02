import random
import numpy as np
from operator import methodcaller
import matplotlib.pyplot as plt

class City:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def distance(self, city):
        distX = abs(self.x - city.x)
        distY = abs(self.y - city.y)
        distance = np.sqrt((distX ** 2) + (distY ** 2))
        return distance
    def __str__(self):
        return "city: x="+str(self.x)+" y="+str(self.y)
    def __eq__(self, value):
        if self.x==value.x and self.y==value.y:
            return True
        return False

class Chromosome:
    def  __init__(self,cities,shaffel):
        if (shaffel):
            random.shuffle(cities)
            self.cities=cities
            x=0
        else:
            self.cities=cities
        self.chromosomeSize=len(self.cities)

    def CheckSameCity(self):
        for i in range(len(self.cities)):
            for j in range(len(self.cities)):
                city1=self.cities[i]
                city2=self.cities[j]
                if (i!=j and city1==city2):
                    print("Check bad")
                    print(city1)
                    print(city2)
                    return False
        return True

    def Plot(self):
        lstX=[]
        lstY=[]
        for city in self.cities:
            lstX.append(city.x)
            lstY.append(city.y)
        plt.scatter(lstX,lstY)
        plt.show()

    def PlotLines(self):
        lstX=[]
        lstY=[]
        for city in self.cities:
            lstX.append(city.x)
            lstY.append(city.y)
        plt.scatter(lstX,lstY)
        for i in range(len(self.cities)-1):
            lx=[]
            ly=[]
            lx.append(self.cities[i].x)
            lx.append(self.cities[i+1].x)
            ly.append(self.cities[i].y)
            ly.append(self.cities[i+1].y)
            plt.plot(lx,ly,'k-')
        plt.plot([self.cities[0].x,self.cities[len(self.cities)-1].x],[self.cities[0].y,self.cities[len(self.cities)-1].y],'k-')
        plt.axis('equal')
        plt.show()

    def printCities(self):
        for city in self.cities:
            print(city)

    def distance(self):
        sum=0
        for i in range(len(self.cities)-2):
            sum+=self.cities[i].distance(self.cities[i+1])
        sum+=self.cities[0].distance(self.cities[len(self.cities)-1])
        return sum
    def getCities(self):
        return self.cities

    def mutate(self):
        i=0
        j=0
        while(i==j):
            i=int(random.random()*self.chromosomeSize);
            j=int(random.random()*self.chromosomeSize);
            city=self.cities[i]
            self.cities[i]=self.cities[j]
            self.cities[j]=city
    
class Population:
      def __init__(self,populationSize,numOfCitiesInEachChromosome,elitismPercent,parentsPercent,mutationPercent):
          self.elitismPercent=elitismPercent
          self.chromosomeSize=numOfCitiesInEachChromosome
          self.populationSize=populationSize
          self.parentsPercent=parentsPercent
          self.mutationPercent=mutationPercent
          self.sortedChromosomes=[]
          self.chromosomes=[]
          self.cities = []
          for i in range(0,self.chromosomeSize):
            self.cities.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))
        
          for i in range(populationSize):
              self.chromosomes.append(Chromosome(self.cities,True))
          self.sort()

      def Plot(self):
          self.chromosomes[0].Plot()
  
      def distance(city1,city2):
          return city1.distance(city2)

      def sort(self):
          self.sortedChromosomes=sorted(self.chromosomes,key=methodcaller('distance'))

      def PrintDistance(self):
           print(self.chromosomes[0].distance())
      def GetDistance(self):
          return self.chromosomes[0].distance()

      def PrintBest(self):
          self.chromosomes[0].printCities()
        
      def PlotBest(self):
          self.chromosomes[0].PlotLines()

      def Print(self):
          i=0
          for chromosome in  self.sortedChromosomes:
              i+=1
              print("chromosome "+str(i))
              chromosome.printCities()
              print(chromosome.distance())
      def crossover(self,chromosome1,chromosome2):
            length=self.chromosomeSize
            start=int(random.random()*length)
            end=int(random.random()*length)
            if (start>end):
                tmp=start
                start=end
                end=tmp
            if (start==end and end==length-1):
                start-=1
            elif (start==end and end<length-1):
                end+=1
          
            citiesx=chromosome1.getCities()
            citiesy=chromosome2.getCities()
           
            childMiddle=[]
            for i in range(start,end):
                childMiddle.append(citiesy[i])

            cities=[]
            place=0
            for i in range(start):
                while (citiesx[place] in childMiddle):
                    place+=1
                cities.append(citiesx[place])
                place+=1
            for i in range(start,end):
                cities.append(citiesy[i])
            
            for i in range(end,length):
                while (citiesx[place] in childMiddle):
                    place+=1
                cities.append(citiesx[place])
                place+=1
            child=Chromosome(cities,False)
            ret3=child.CheckSameCity()
            return child

      def PrepareNextGeneration(self):
          NextGeneration=[]
          elitism=int(self.elitismPercent*self.populationSize)
          for k in range(elitism):
              NextGeneration.append(self.sortedChromosomes[k])

          m=int(self.populationSize*self.parentsPercent)
          selectedParents=self.sortedChromosomes[:m]
          while len(NextGeneration)<self.populationSize:
            i=0
            j=0
            while(i==j):
                i=int(random.random()*m)
                j=int(random.random()*m)
            parent1=selectedParents[i]
            parent2=selectedParents[j]
            child=self.crossover(parent1,parent2)
            x=random.random()
            if (x<=self.mutationPercent):
                child.mutate()
            NextGeneration.append(child)
           
          self.chromosomes=NextGeneration
          self.sort()

population=Population(100,25,0.02,0.2,0.02)
population.Plot()


y=[]
x=[]
for i in range(500):
    population.PrepareNextGeneration()
    population.PrintDistance()
    y.append(population.GetDistance())
    x.append(i)

plt.plot(x,y)
plt.show()

population.PrintBest()
population.PlotBest()






