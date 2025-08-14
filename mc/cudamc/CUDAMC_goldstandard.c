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

#define SIGN(x) ((x)>=0 ? 1:-1)


void MCh(unsigned int* xd,unsigned int* cd, unsigned int* ad,unsigned int* numh,unsigned int* histh)
{
	unsigned int block;
	unsigned int thread;
	float3 pos; //float triplet to store the position
	float3 dir; //float triplet to store the direction
	float t;	//float to store the time of flight
	float s;	//step length
	
	float mus_max=90.0f;	//[1/cm]
	float v=0.0214f;		//[cm/ps] (c=0.03 [cm/ps] v=c/n) here n=1.4
	float cos_crit=0.6999f;	//the critical angle for total internal reflection at the border cos_crit=sqrt(1-(nt/ni)^2)
	float g=0.9f;	
	float n=1.4f;

	unsigned int ii;
	unsigned long long int x;
	unsigned int a;
	unsigned int num_det_photons;
	unsigned int flag;


	for(block=0;block<NUM_BLOCKS;block++)
	{
		//printf("\nblock=%d ",block);
		for(thread=0;thread<NUM_THREADS_PER_BLOCK;thread++)
		{
		
		x=cd[NUM_THREADS_PER_BLOCK*block+thread];
		x=(x<<32)+xd[NUM_THREADS_PER_BLOCK*block+thread];

		a=ad[NUM_THREADS_PER_BLOCK*block+thread];

		num_det_photons=0;
		flag=0;



		LaunchPhotonh(&pos, &dir, &t);//Launch the photon

		for(ii=0;ii<NUMSTEPS_CPU;ii++) //this is the main while loop
		{
			s = -logf(rand_MWC_och(&x,&a))/mus_max;//sample step length 
			
			//Perform boundary crossing check here
			if((pos.z+dir.z*s)<=0)//photon crosses boundary within the next step
			{
				flag=Reflecth(&dir,&pos,&t,&v,&cos_crit,&n,&x,&a,histh);
			}

			//Move (we can move the photons that have been terminated above since it improves our performance and does not affect our results)
			pos.x += s*dir.x;
			pos.y += s*dir.y;
			pos.z += s*dir.z;
			t += s/v; 

			Spinh(&dir,&g,&x,&a);

			if(t >= TMAX || flag>=1)//Kill photon and launch a new one
			{
				num_det_photons++;
				flag=0;
				LaunchPhotonh(&pos, &dir, &t);//Launch the photon
			}
			

		}//end main for loop!
		
		numh[NUM_THREADS_PER_BLOCK*block+thread]=num_det_photons; 

		}//end for thread
	}//end for block
}//end MCh

float rand_MWC_coh(unsigned long long* x, unsigned int* a)
{
		//this implementation seems to be faster.
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return((float)((unsigned int)(*x&0xffffffffull))/(UINT_MAX));

}//end __device__ rand_MWC_coh



float rand_MWC_och(unsigned long long* x, unsigned int* a)
{
		*x=(*x&0xffffffffull)*(*a)+(*x>>32);
		return(1.0f-(float)((unsigned int)(*x&0xffffffffull))/(UINT_MAX));//return(1.0f-(float)((unsigned int)(*x&0xffffffffull))/(UINT_MAX));
}//end __device__ rand_MWC_och

void LaunchPhotonh(float3* pos, float3* dir, float* t)
{
	pos->x=0.0f;
	pos->y=0.0f;
	pos->z=0.0f;

	dir->x=0.0f;
	dir->y=0.0f;
	dir->z=1.0f;

	*t=0.0f;
}



void Spinh(float3* dir, float* g, unsigned long long* x, unsigned int* a)
{
	float cost, sint;	// cosine and sine of the 
						// polar deflection angle theta. 
	float cosp, sinp;	// cosine and sine of the 
						// azimuthal angle psi. 
	float temp;

	float tempdir=dir->x;

	if((*g)==0.0f)
		cost = 2.0f*rand_MWC_coh(x,a) -1.0f;//Should be close close??!!!!!
	else
	{
		temp = (1.0f-(*g)*(*g))/(1.0f-(*g)+2.0f*(*g)*rand_MWC_coh(x,a));//Should be close close????!!!!!
		cost = (1.0f+(*g)*(*g) - temp*temp)/(2.0f*(*g));
	}
	sint = sqrtf(1.0f - cost*cost);

	temp = 2.0f*PI*rand_MWC_coh(x,a); // spin psi [0-2*PI)
	cosp = cosf(temp);
	sinp = sinf(temp);
	
	temp = sqrtf(1.0f - dir->z*dir->z);
	if(temp==0.0f)// normal incident.
	{
		dir->x = sint*cosp;
		dir->y = sint*sinp;
		dir->z = copysignf(cost,dir->z*cost);
		
	}
	else // regular incident.
	{
		dir->x = sint*(dir->x*dir->z*cosp - dir->y*sinp)/temp + dir->x*cost;
		dir->y = sint*(dir->y*dir->z*cosp + tempdir*sinp)/temp + dir->y*cost;
		dir->z = -sint*cosp*temp + dir->z*cost;
	}

	//normalisation seems to be required as we are using floats! Otherwise the small numerical error will accumulate
	temp=rsqrtf(dir->x*dir->x+dir->y*dir->y+dir->z*dir->z);
	dir->x=dir->x*temp;
	dir->y=dir->y*temp;
	dir->z=dir->z*temp;
	

}


unsigned int Reflecth(float3* dir, float3* pos, float* t, float* v, float* cos_crit, float* n, unsigned long long* x, unsigned int* a,unsigned int* histh)
{
	float r;
	float fibre_separtion=1.0f;//[cm]
	float fibre_diameter=0.05f;//[cm]
	unsigned int adr;


	if(-dir->z<=*cos_crit)
		r=1.0f; //total internal reflection
	else
	{
		if(-dir->z==1.0f)//normal incident
		{		
			r = (1-*n)/(1+*n);
			r *= r;//square
		}
		else
		{
			//long and boring calculations of r
			float sinangle_i = sqrtf(1-dir->z*dir->z);
			float sinangle_t = *n*sinangle_i;
			float cosangle_t = sqrtf(1-sinangle_t*sinangle_t);
			
			float cossumangle = (-dir->z*cosangle_t) - sinangle_i*sinangle_t;
			float cosdiffangle = (-dir->z*cosangle_t) + sinangle_i*sinangle_t;
			float sinsumangle = sinangle_i*cosangle_t + (-dir->z*sinangle_t);
			float sindiffangle = sinangle_i*cosangle_t - (-dir->z*sinangle_t); 
			
			r = 0.5*sindiffangle*sindiffangle*(cosdiffangle*cosdiffangle+cossumangle*cossumangle)/(sinsumangle*sinsumangle*cosdiffangle*cosdiffangle);
		}
	}
	if(r<1.0f)
	{
		if(rand_MWC_coh(x,a)<=r)//reflect
			r=1.0f;
		else//transmitt
		{
			//calculate x and y where the photon escapes the medium
			
			r=-(pos->z/dir->z);//dir->z must be finite since we have a boundary cross!
			pos->x+=dir->x*r;
			pos->y+=dir->y*r;
			*t+=r/(*v); //calculate the time when the photon exits

			r=sqrtf(pos->x*pos->x+pos->y*pos->y);
			
			//check for detection here
			if(fabsf(r-fibre_separtion)<=fibre_diameter)//!!!!!!!!!!!!!!!!!!!!!!!
			{
				adr=(unsigned int)floorf((*t)/DT);
				//photon detected!
				histh[adr]=histh[adr]+1;
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

