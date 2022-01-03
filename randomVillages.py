import random

number = random.randint(500,500)
rang = random.randint(1,50)
Vc = random.randint(1,4)
Fc = random.randint(1,100)
capacity = random.randint(5,20)
maxX = 200
maxY = 200
maxC = 20

with open('minimart-'+str(number)+'.dat', 'w') as f:
    f.write('param n :=    '+str(number)+'; \n')			
    f.write('param range :=    '+str(rang)+'; \n')
    f.write('param Vc :=    '+str(Vc)+'; \n')
    f.write('param Fc :=	'+str(Fc)+'	; \n')			
    f.write('param capacity :='+str(capacity)+";\n")
    f.write('param:	Cx  Cy	Dc	usable	:= \n')
    for i in range(1,number+1):
        x = random.randint(1,maxX)
        y = random.randint(1,maxY)
        dc = random.randint(1,maxC)
        isOpen = random.randint(0,1)
        f.write(str(i)+'    '+str(x)+'    '+str(y)+'    '+str(dc)+'   '+str(isOpen)+'\n')

    f.write('                           ;')