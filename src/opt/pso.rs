use std;
use std::ops::IndexMut;

use rand::Rng;
use rand::distributions::range::SampleRange;
use num_traits::cast::NumCast;
use num_traits::float::Float;
use num_traits::identities::{one, zero};
use std::fmt::Debug;
use super::super::utils::{InitFromLen};




#[derive(Debug, Clone)]
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

pub struct ParticleSwarmMaximizer<V,T>
where T: Float + NumCast + std::cmp::PartialOrd + Copy+Default +  SampleRange+Debug+Send+Sync,
      V: Clone + IndexMut<usize, Output = T> + InitFromLen +  Debug+Send+Sync,
{
    pub particle_count:usize,
    pub ndim:usize,
    pub swarm:Vec<Particle<V,T> >,
    pub gbest:Option<Particle<V,T> >,
    pub func:Box<Fn(&V)->T+Send+Sync>
}

impl<V,T> ParticleSwarmMaximizer<V,T>
where T: Float + NumCast + std::cmp::PartialOrd + Copy +Default+  SampleRange+Debug+Send+Sync,
      V: Clone + IndexMut<usize, Output = T> + InitFromLen + Debug+Send+Sync,
{
    pub fn new<R>(func:Box<Fn(&V)->T+Send+Sync>, lower:V, upper:V, guess:Option<V>, particle_count:usize, rng:&mut R)->ParticleSwarmMaximizer<V,T>
    where R:Rng
    {
        let swarm=Self::init_swarm(&func,&lower, &upper, particle_count, rng);
        let ndim=lower.len();
        let gbest=guess.map(|p|{
            let f=func(&p);
            Particle{position:p, velocity:V::init(ndim), fitness:f,pbest:None}
        });
        ParticleSwarmMaximizer{
            particle_count:particle_count,
            ndim:ndim, swarm:swarm, gbest:gbest,
            func:func
        }
    }

    pub fn restart<R>(&mut self, lower:V, upper:V, particle_count:usize, rng:&mut R)
    where R:Rng
    {
        self.swarm=Self::init_swarm(&self.func, &lower, &upper, particle_count, rng);
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

    pub fn converged(&self, p:T, m1:T, m2:T)->bool{
        self.converge_dfit(p, m1) && self.converge_dspace(p, m2)
    }

    pub fn converge_dfit(&self,p:T,  m:T)->bool{
        let mut best_sort:Vec<T>=self.swarm.iter().map(|x|{x.fitness}).collect();
        best_sort.sort_unstable_by(|&a,&b|{
            if a>b{
                std::cmp::Ordering::Less
            }
            else if a<b{
                std::cmp::Ordering::Greater
            }
            else{
                std::cmp::Ordering::Equal
            }
        });
        let i1:usize=(T::from(self.particle_count).unwrap()*p).floor().to_usize().unwrap();
        let best_mean=best_sort[1..i1].iter().fold(zero::<T>(), |a,&b|{a+b})/NumCast::from(i1-1).unwrap();
        if let Some(ref x)=self.gbest{
            (x.fitness-best_mean).abs()<m
        }else{
            false
        }
    }

    pub fn converge_dspace(&self, p:T, m:T)->bool{
        let mut sorted_swarm:Vec<_>=self.swarm.iter().map(|x|{x.clone()}).collect();
        sorted_swarm.sort_unstable_by(|a, b|{
            if a.fitness>b.fitness{
                std::cmp::Ordering::Less
            }else if a.fitness<b.fitness{
                std::cmp::Ordering::Greater
            }else{
                std::cmp::Ordering::Equal
            }
        });
        let i1:usize=(T::from(self.particle_count).unwrap()*p).floor().to_usize().unwrap();
        if let Some(ref gbest)=self.gbest{
            let max_norm:T=
            sorted_swarm[0..i1].iter().map(|x|{
                let mut diff_norm:T=zero();

                for i in 0..self.ndim{
                    diff_norm=diff_norm+(gbest.position[i]-x.position[i]).powi(2);
                }
                diff_norm
            }).fold(zero::<T>(), |a:T,b:T| b.max(a));
            max_norm<m
        }else{
            false
        }
    }
}
