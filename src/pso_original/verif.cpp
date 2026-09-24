#include <cstdio>
#include <cstdlib>
#include <cmath>
float getRandom(float low, float high){ return low + float(((high-low)+1)*rand()/(RAND_MAX+1.0)); } // version du prof
int main(){
  float mn=1e9f, mx=-1e9f;
  for(int i=0;i<1000000;i++){ float r=getRandom(-5.12f,5.12f); mn=fminf(mn,r); mx=fmaxf(mx,r); }
  printf("[Bug 3] getRandom(-5.12,5.12) donne des valeurs dans [%f, %f]\n", mn, mx);

  float x[3]={1.f,2.f,3.f}, somme=0, produit=0;               // version du prof
  for(int i=0;i<3;i++){ somme+=x[i]*x[i]/4000; produit*=cosf(x[i]/sqrtf(i+1.f)); }
  float p1=1; for(int i=0;i<3;i++) p1*=cosf(x[i]/sqrtf(i+1.f)); // version correcte
  printf("[Bug 2] produit Griewank : prof = %f, correct = %f\n", produit, p1);

  printf("[Bug 9] pas du float autour de 450 : %g\n", nextafterf(450.f,1e9f)-450.f);
  printf("[Bug 7] FE réalisées = 30000 x 512 = %ld  vs budget = 30000\n", 30000L*512);
}
