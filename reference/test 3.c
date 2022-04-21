#include <stdio.h>
#include <math.h>

int 
main(){
    float h = 6.626 * pow(10,(-34));
    float L = 3*pow(10,(-9));
    float r = h/(3*1.505/(2*L*2*L));
    printf(r);
}