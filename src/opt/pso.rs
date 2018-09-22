use std;
use std::ops::IndexMut;

use rand::Rng;
use rand::distributions::range::SampleRange;
use num_traits::cast::NumCast;
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use std::fmt::Debug;
use super::super::utils::{InitFromLen};




#[derive(Debug)]
pub struct Particle<V,T>
where
      T: Float + NumCast + std::cmp::PartialOrd + Copy +Default+ SampleRange+ Debug+Send+Sync,
      V: Clone + IndexMut<usize, Output = T> + InitFromLen+ Debug+Send+Sync,

{
    pub position:V,
    pub velocity:V,
    pub fitness:T,
    pub pbest: Option<Box<Particle<V,T>>>
}

pub struct ParticleSwarmOptimizer<V,T>
where T: Float + NumCast + std::cmp::PartialOrd + Copy+Default +  SampleRange+Debug+Send+Sync,
      V: Clone + IndexMut<usize, Output = T> + InitFromLen +  Debug+Send+Sync,
{
    pub lower:V,
    pub upper:V,
    pub particle_count:usize,
    pub ndim:usize,
    pub swarm:Vec<Particle<V,T> >,
    pub gbest:Option<Particle<V,T> >,
    pub func:Box<Fn(&V)->T+Send+Sync>
}

impl<V,T> ParticleSwarmOptimizer<V,T>
where T: Float + NumCast + std::cmp::PartialOrd + Copy +Default+  SampleRange+Debug+Send+Sync,
      V: Clone + IndexMut<usize, Output = T> + InitFromLen + Debug+Send+Sync,
{
    pub fn new<R>(func:Box<Fn(&V)->T+Send+Sync>, lower:V, upper:V, particle_count:usize, rng:&mut R)->ParticleSwarmOptimizer<V,T>
    where R:Rng
    {
        let swarm=Self::init_swarm(&func,&lower, &upper, particle_count, rng);
        let ndim=lower.len();
        ParticleSwarmOptimizer{
            lower:lower, upper:upper, particle_count:particle_count, 
            ndim:ndim, swarm:swarm, gbest:None,
            func:func
        }
    }

    pub fn init_swarm<R>(func:&Box<Fn(&V)->T+Sync+Send>, lower:&V, upper:&V, pc:usize, rng:&mut R)->Vec<Particle<V,T>>
    where R:Rng
    {
        let mut result=Vec::<Particle<V,T> >::new();
        let ndim=lower.len();
        for _i in 0..pc{
            let mut p=V::init(ndim);
            let mut v=V::init(ndim);
            for j in 0..ndim{
                p[j]=rng.gen_range(lower[j], upper[j]);
                v[j]=zero();
            }
            let f=func(&p);
            result.push(Particle{position:p, velocity:v, fitness:f, pbest:None});
        }
        result
    }

    pub fn update_fitness(&mut self){
        let f:Vec<T>=self.swarm.iter().map(|p|{
            (self.func)(&p.position)
        }).collect();
        f.iter().zip(self.swarm.iter_mut()).for_each(|(&f,p)|{
            p.fitness=f;
        });
    }

    pub fn sample<R>(&mut self, rng:&mut R, c1:T, c2:T)
    where R:Rng
    {
        for p in &mut self.swarm{
            match self.gbest{
                None=>{self.gbest=Some(Particle{
                    position:p.position.clone(),
                    velocity:p.velocity.clone(),
                    fitness:p.fitness,
                    pbest:None,
                })},
                Some(ref mut gb) => {
                    if gb.fitness<p.fitness{
                        gb.position=p.position.clone();
                        gb.velocity=p.velocity.clone();
                        gb.fitness=p.fitness;
                    }
                }

            }

            match p.pbest{
                None => p.pbest=Some(Box::new(Particle{
                    position:p.position.clone(),
                    velocity:p.velocity.clone(),
                    fitness:p.fitness,
                    pbest:None,
                })),

                Some(ref mut pb)=>{
                    if p.fitness > pb.fitness{
                        pb.position=p.position.clone();
                        pb.velocity=p.velocity.clone();
                        pb.fitness=p.fitness;
                    }
                }
            }
        }

        for p in &mut self.swarm{
            if let Some(ref pbest)=p.pbest{
                if let Some(ref gbest)=self.gbest{
                    for j in 0..self.ndim{
                        let w=(one::<T>()+rng.gen_range(zero(), one()))/(one::<T>()+one::<T>());
                        let part_vel=w*p.velocity[j];
                        let cog_vel=c1*rng.gen_range(zero(), one())*(pbest.position[j]-p.position[j]);
                        let soc_vel=c2*rng.gen_range(zero(), one())*(gbest.position[j]-p.position[j]);
                        p.velocity[j]=part_vel+cog_vel+soc_vel;
                        p.position[j]=p.position[j]+p.velocity[j]
                    }
                }
            }
        }

        self.update_fitness();
    }
}
