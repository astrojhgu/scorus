use std::collections::BTreeMap;

use crate::coordinates::Vec3d;
use num::traits::float::Float;

pub struct Tessellation<T>
where
    T: Float + Copy,
{
    pub vertices: Vec<Vec3d<T>>,
    pub faces: Vec<[usize; 3]>,
}

fn mid_point<T>(p1: &Vec3d<T>, p2: &Vec3d<T>) -> Vec3d<T>
where
    T: Float + Copy,
{
    let two = T::one() + T::one();
    Vec3d::new(
        (p1.x + p2.x) / two,
        (p1.y + p2.y) / two,
        (p1.z + p2.z) / two,
    )
}

fn normalized_mid_point<T>(p1: &Vec3d<T>, p2: &Vec3d<T>) -> Vec3d<T>
where
    T: Float + Copy,
{
    mid_point(p1, p2).normalized()
}

fn regulate(mut v: [usize; 3]) -> [usize; 3] {
    v.sort();
    v
}

impl<T> Tessellation<T>
where
    T: Float + Copy,
{
    pub fn octahedron() -> Tessellation<T> {
        let one = T::one();
        let zero = T::zero();
        let vertices = vec![
            Vec3d::new(zero, zero, one),
            Vec3d::new(one, zero, zero),
            Vec3d::new(zero, one, zero),
            Vec3d::new(-one, zero, zero),
            Vec3d::new(zero, -one, zero),
            Vec3d::new(zero, zero, -one),
        ];
        let faces = vec![
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [1, 0, 4],
            [2, 1, 5],
            [3, 2, 5],
            [4, 3, 5],
            [1, 4, 5],
        ];
        Tessellation { vertices, faces }
    }

    pub fn northern_hemispere() -> Tessellation<T> {
        let one = T::one();
        let zero = T::zero();
        let vertices = vec![
            Vec3d::new(zero, zero, one),
            Vec3d::new(one, zero, zero),
            Vec3d::new(zero, one, zero),
            Vec3d::new(-one, zero, zero),
            Vec3d::new(zero, -one, zero),
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]];
        Tessellation { vertices, faces }
    }

    pub fn refine(&mut self) {
        let num_of_points_init = self.vertices.len();
        for f in &mut self.faces {
            *f = regulate(*f);
        }

        let center = self
            .vertices
            .iter()
            .fold(Vec3d::new(T::zero(), T::zero(), T::zero()), |a, &b| a + b)
            / T::from(num_of_points_init).unwrap();

        for p in &mut self.vertices {
            *p = *p - center;
            *p = p.normalized();
        }

        let mut edge_mid_map = BTreeMap::<(usize, usize), usize>::new();
        let mut new_faces = Vec::<[usize; 3]>::new();

        for f in &self.faces {
            let edges = [(f[0], f[1]), (f[1], f[2]), (f[0], f[2])];

            let mut new_vertex_ids = [0, 0, 0];
            for i in 0..3 {
                match edge_mid_map.get(&(edges[i])) {
                    Some(&id) => {
                        new_vertex_ids[i] = id;
                    }
                    _ => {
                        let mid = normalized_mid_point(
                            &self.vertices[edges[i].0],
                            &self.vertices[edges[i].1],
                        );
                        new_vertex_ids[i] = self.vertices.len();
                        edge_mid_map.insert((edges[i].0, edges[i].1), new_vertex_ids[i]);
                        self.vertices.push(mid);
                    }
                }
            }

            new_faces.push(regulate([new_vertex_ids[0], new_vertex_ids[1], f[1]]));
            new_faces.push(regulate([new_vertex_ids[1], new_vertex_ids[2], f[2]]));
            new_faces.push(regulate([new_vertex_ids[2], new_vertex_ids[0], f[0]]));
            new_faces.push(regulate([
                new_vertex_ids[0],
                new_vertex_ids[1],
                new_vertex_ids[2],
            ]));
        }

        for p in &mut self.vertices {
            *p = *p + center;
        }
        self.faces = new_faces;
    }

    pub fn regulate_norm(&mut self) {
        self.faces.iter_mut().for_each(|v| {
            let v0 = self.vertices[v[0]];
            let v1 = self.vertices[v[2]];
            let v2 = self.vertices[v[2]];
            let x1 = v1 - v0;
            let x2 = v2 - v0;
            let norm = x1.cross(x2);
            let w = x1 + x2 + x2;
            if norm.dot(w) < T::zero() {
                v.swap(1, 2);
            }
        });
    }
}
