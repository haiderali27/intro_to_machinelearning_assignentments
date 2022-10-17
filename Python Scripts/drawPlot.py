import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np
#z = np.linspace(0,2.0,10)
np.random.seed(42)
x = []
y = []





print(np.multiply(np.array([1,2,3]), np.array([4,5,6])))


def mylinfit(x,y):
  a=0
  b = 0
  N=len(y)
  numpy_x= np.array(x)
  numpy_y= np.array(y)
  sum_x=np.sum(numpy_x)
  sum_y=np.sum(numpy_y)
  sum_x_square=np.sum(numpy_x**2)
  sum_y_square=np.sum(numpy_y**2)
  print("=================",numpy_x, numpy_y, sum_x, sum_y, sum_x_square, sum_y_square, np.multiply(x, y), np.sum(np.multiply(x, y)), np.sum(sum_y*numpy_x), sum_y*numpy_x,"===============")

  a= ((N*sum_y*sum_x)-np.sum(sum_y*numpy_x))/((N*sum_x_square)-(sum_x**2))
  b= (sum_y-(a*sum_x))/N
  return a,b


x1=[5,6,7]
y1=[1,2,3]

print(mylinfit(x1,y1))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])

def onclick(event):
   #global x,y
   if event.button is MouseButton.LEFT:
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %(event.button, event.x, event.y, event.xdata, event.ydata))
    plt.plot(event.xdata, event.ydata, 'bo')
    fig.canvas.draw()
    x.append(event.xdata)
    y.append(event.ydata)
    #print(x, y)
    #plt.clf()
    plt.plot(x, y,'bo')
    plt.xlabel('x')
    plt.ylabel('y')
    fig.canvas.draw()
   else:
    print("Right Click")
    a,b=mylinfit(x,y)
    #print("a:",a,", b:",b)
    slope_y= lambda x: a*np.array(x)+b
    #plt.plot(x, slope_y)
    #plt.plot(x,slope_y(x),label=f'a={a} b={b}')
    z = np.polyfit(x, y, 3)
    plt.plot(x,z)
    plt.legend()
    fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)




plt.plot(x, y,'mo')
plt.show()




