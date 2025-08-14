/*	This file is part of CUDAMC.

    CUDAMC is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMC is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMC.  If not, see <http://www.gnu.org/licenses/>.*/

__global__ void MCd(unsigned int* xd,unsigned int* cd, unsigned int* ad,unsigned int* numd,unsigned int* histd)
{

    //for loops
    unsigned int ii=0;

    //First element processed by the block
    //int begin=NUM_THREADS_PER_BLOCK*bx;

    unsigned long long int x=cd[NUM_THREADS_PER_BLOCK*blockIdx.x+threadIdx.x];
	
	x=(x<<32)+xd[NUM_THREADS_PER_BLOCK*blockIdx.x+threadIdx.x];


    unsigned int a=ad[NUM_THREADS_PER_BLOCK*blockIdx.x+threadIdx.x];


	float3 pos; //float triplet to store the position
	float3 dir; //float triplet to store the direction
	float t;	//float to store the time of flight
	float s;	//step length
	
	float mus_max=90.0f;	//[1/cm]
	float v=0.0214f;		//[cm/ps] (c=0.03 [cm/ps] v=c/n) here n=1.4
	float cos_crit=0.6999f;	//the critical angle for total internal reflection at the border cos_crit=sqrt(1-(nt/ni)^2)
	float g=0.9f;	
	float n=1.4f;

	unsigned int num_det_photons=0;
	unsigned int flag=0;

	LaunchPhoton(&pos, &dir, &t);//Launch the photon
	
	for(ii=0;ii<NUMSTEPS_GPU;ii++) //this is the main while loop
	{
		//num_det_photons++;
		s = __fdividef(-__logf(rand_MWC_oc(&x,&a)),mus_max);//sample step length 
		
		//Perform boundary crossing check here
		if((pos.z+dir.z*s)<=0)//photon crosses boundary within the next step
		{
			flag=Reflect(&dir,&pos,&t,&v,&cos_crit,&n,&x,&a,histd);
		}
		
		//Move (we can move the photons that have been terminated above since it improves our performance and does not affect our results)
		pos.x += s*dir.x;
		pos.y += s*dir.y;
		pos.z += s*dir.z;
		t += __fdividef(s,v); 

		Spin(&dir,&g,&x,&a);

		if(t >= TMAX || flag>=1)//Kill photon and launch a new one
		{
			num_det_photons++;
			flag=0;
			LaunchPhoton(&pos, &dir, &t);//Launch the photon
		}
		

	}//end main for loop!
	

	__syncthreads();//necessary?

	numd[NUM_THREADS_PER_BLOCK*blockIdx.x+threadIdx.x]/*[begin+tx]*/=num_det_photons; 

}//end MCd

__device__ float rand_MWC_co(unsigned long long* x,//unsigned int* c,
		       unsigned int* a)
{
		//Generate a random number [0,1)
		//this implementation seems to be faster
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return((float)((unsigned int)(*x&0xffffffffull))/(UINT_MAX));

}//end __device__ rand_MWC_co

__device__ float rand_MWC_oc(unsigned long long* x,//unsigned int* c,
		       unsigned int* a)
{
		//Generate a random number (0,1]
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return(1.0f-(float)((unsigned int)(*x&0xffffffffull))/(UINT_MAX));
}//end __device__ rand_MWC_oc


__device__ void LaunchPhoton(float3* pos, float3* dir, float* t)
{
	pos->x=0.0f;
	pos->y=0.0f;
	pos->z=0.0f;

	dir->x=0.0f;
	dir->y=0.0f;
	dir->z=1.0f;

	*t=0.0f;
}



__device__ void Spin(float3* dir, float* g, unsigned long long* x,//unsigned int* c,
		       unsigned int* a)
{
	float cost, sint;	// cosine and sine of the 
						// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
						// azimuthal angle psi. 
	float temp;

	float tempdir=dir->x;


	//This is more efficient for g!=0 but of course less efficient for g==0
	temp = __fdividef((1.0f-(*g)*(*g)),(1.0f-(*g)+2.0f*(*g)*rand_MWC_co(x,a)));//Should be close close????!!!!!
	cost = __fdividef((1.0f+(*g)*(*g) - temp*temp),(2.0f*(*g)));
	if((*g)==0.0f)
		cost = 2.0f*rand_MWC_co(x,a) -1.0f;


	sint = sqrtf(1.0f - cost*cost);

	__sincosf(2.0f*PI*rand_MWC_co(x,a),&cosp,&sinp);
	

	temp = sqrtf(1.0f - dir->z*dir->z);

	if(temp==0.0f)// normal incident.
	{
		dir->x = sint*cosp;
		dir->y = sint*sinp;
		dir->z = copysignf(cost,dir->z*cost);
	}
	else // regular incident.
	{
		dir->x = __fdividef(sint*(dir->x*dir->z*cosp - dir->y*sinp),temp) + dir->x*cost;
		dir->y = __fdividef(sint*(dir->y*dir->z*cosp + tempdir*sinp),temp) + dir->y*cost;
		dir->z = -sint*cosp*temp + dir->z*cost;
	}

	//normalisation seems to be required as we are using floats! Otherwise the small numerical error will accumulate
	temp=rsqrtf(dir->x*dir->x+dir->y*dir->y+dir->z*dir->z);
	dir->x=dir->x*temp;
	dir->y=dir->y*temp;
	dir->z=dir->z*temp;
	
}


__device__ unsigned int Reflect(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, unsigned long long* x,//unsigned int* c,
		       unsigned int* a,unsigned int* histd)
{
	float r;
	float fibre_separtion=1.0f;//[cm]
	float fibre_diameter=0.05f;//[cm]

	if(-dir->z<=*cos_crit)
		r=1.0f; //total internal reflection
	else
	{
		if(-dir->z==1.0f)//normal incident
		{		
			r = __fdividef((1.0f-*n),(1+*n));
			r *= r;//square
		}
		else
		{
			//long and boring calculations of r
			float sinangle_i = sqrtf(1.0f-dir->z*dir->z);
			float sinangle_t = *n*sinangle_i;
			float cosangle_t = sqrtf(1.0f-sinangle_t*sinangle_t);
			
			float cossumangle = (-dir->z*cosangle_t) - sinangle_i*sinangle_t;
			float cosdiffangle = (-dir->z*cosangle_t) + sinangle_i*sinangle_t;
			float sinsumangle = sinangle_i*cosangle_t + (-dir->z*sinangle_t);
			float sindiffangle = sinangle_i*cosangle_t - (-dir->z*sinangle_t); 
			
			r = 0.5*sindiffangle*sindiffangle*__fdividef((cosdiffangle*cosdiffangle+cossumangle*cossumangle),(sinsumangle*sinsumangle*cosdiffangle*cosdiffangle));
		
		}
	}
	if(r<1.0f)
	{
		if(rand_MWC_co(x/*,c*/,a)<=r)//reflect
			r=1.0f;
		else//transmitt
		{
			//calculate x and y where the photon escapes the medium
			
			r=__fdividef(pos->z,-dir->z);//dir->z must be finite since we have a boundary cross!
			pos->x+=dir->x*r;
			pos->y+=dir->y*r;
			*t+=__fdividef(r,*v); //calculate the time when the photon exits

			r=sqrtf(pos->x*pos->x+pos->y*pos->y);
			
			//check for detection here
			if(fabsf(r-fibre_separtion)<=fibre_diameter)
			{
				//photon detected!
-				atomicAdd( histd + __float2uint_rz(__fdividef((*t),DT)) , 1);//&histd[(unsigned int)floorf(__fdividef((t*),DT))],(unsigned int)1);
				return 1;
			}
			else
			{
				return 2;
			}	
		}
	}
	if(r==1.0f)//reflect (mirror z and dz in reflection plane)
	{
		pos->z *= -1;//mirror the z-coordinate in the z=0 plane, equal to a reflection.
		dir->z *= -1;// do the same to the z direction vector
	}
	return 0;
}
