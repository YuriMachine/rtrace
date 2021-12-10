
extern crate nalgebra_glm as glm;
extern crate rand;

#[allow(dead_code)]
mod components;
#[allow(dead_code)]
mod trace_utils;
mod trace;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
