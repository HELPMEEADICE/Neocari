struct Params { multiply: vec4<f32>, screen: vec4<f32>, modes: vec4<u32>, opacity: vec4<f32> }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(1) @binding(0) var backdrop: texture_2d<f32>;
@group(1) @binding(1) var backdrop_sampler: sampler;
@group(2) @binding(0) var mask: texture_2d<f32>;
@group(2) @binding(1) var mask_sampler: sampler;
@group(3) @binding(0) var<uniform> params: Params;
struct Output { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) index: u32) -> Output {
    let positions = array<vec2<f32>,3>(vec2(-1.0,-1.0),vec2(-1.0,3.0),vec2(3.0,-1.0));
    var out: Output;
    out.position = vec4(positions[index],0.0,1.0);
    out.uv = vec2((positions[index].x+1.0)*0.5,(1.0-positions[index].y)*0.5);
    return out;
}
fn luminance(c: vec3<f32>) -> f32 { return dot(c,vec3(0.3,0.59,0.11)); }
fn saturation(c: vec3<f32>) -> f32 { return max(c.x,max(c.y,c.z))-min(c.x,min(c.y,c.z)); }
fn set_luminance(color: vec3<f32>,value: f32) -> vec3<f32> {
    var c = color + vec3(value-luminance(color));
    let low = min(c.x,min(c.y,c.z));
    let high = max(c.x,max(c.y,c.z));
    let l = luminance(c);
    if low < 0.0 { c = vec3(l) + (c-vec3(l))*l/max(l-low,0.000001); }
    if high > 1.0 { c = vec3(l) + (c-vec3(l))*(1.0-l)/max(high-l,0.000001); }
    return c;
}
fn set_saturation(c: vec3<f32>,s: f32) -> vec3<f32> {
    let low = min(c.x,min(c.y,c.z));
    let high = max(c.x,max(c.y,c.z));
    return select(vec3(0.0),(c-vec3(low))*s/max(high-low,0.000001),high>low);
}
fn hard_light(base: vec3<f32>,light: vec3<f32>) -> vec3<f32> {
    return select(vec3(1.0)-2.0*(vec3(1.0)-base)*(vec3(1.0)-light),2.0*base*light,light<=vec3(0.5));
}
fn color_blend(base: vec3<f32>,src: vec3<f32>,mode:u32) -> vec3<f32> {
    switch mode {
        case 3u: { return min(vec3(1.0),base+src); }
        case 4u: { return base+src; }
        case 5u: { return min(base,src); }
        case 6u: { return base*src; }
        case 7u: { return select(vec3(0.0),vec3(1.0)-min(vec3(1.0),(vec3(1.0)-base)/max(src,vec3(0.000001))),src>vec3(0.0)); }
        case 8u: { return max(vec3(0.0),base+src-vec3(1.0)); }
        case 9u: { return max(base,src); }
        case 10u: { return base+src-base*src; }
        case 11u: { return select(vec3(1.0),min(vec3(1.0),base/max(vec3(1.0)-src,vec3(0.000001))),src<vec3(1.0)); }
        case 12u: { return hard_light(src,base); }
        case 13u: {
            let d = select(sqrt(max(base,vec3(0.0))),((16.0*base-vec3(12.0))*base+vec3(4.0))*base,base<=vec3(0.25));
            return select(base+(2.0*src-vec3(1.0))*(d-base),base-(vec3(1.0)-2.0*src)*base*(vec3(1.0)-base),src<=vec3(0.5));
        }
        case 14u: { return hard_light(base,src); }
        case 15u: { return clamp(base+2.0*src-vec3(1.0),vec3(0.0),vec3(1.0)); }
        case 16u: { return set_luminance(set_saturation(src,saturation(base)),luminance(base)); }
        case 17u: { return set_luminance(src,luminance(base)); }
        default: { return src; }
    }
}
@fragment fn fs_composite(input: Output) -> @location(0) vec4<f32> {
    let original = textureSampleLevel(source,source_sampler,input.uv,0.0);
    let dst = textureLoad(backdrop,vec2<i32>(input.position.xy),0);
    var coverage = 1.0;
    if params.modes.z != 0u {
        let m = textureSampleLevel(mask,mask_sampler,input.uv,0.0).r;
        coverage = select(m,1.0-m,params.modes.w!=0u);
    }
    let weight = params.opacity.x*coverage;
    let sa = original.a*weight;
    let da = dst.a;
    let rgb = original.rgb*params.multiply.rgb;
    let colored = (rgb+params.screen.rgb*original.a-rgb*params.screen.rgb)*weight;
    let mode = params.modes.x;
    if mode == 1u { return vec4(colored+dst.rgb,da); }
    if mode == 2u { return vec4(colored*dst.rgb+dst.rgb*(1.0-sa),da); }
    if mode == 0u && params.modes.y == 0u { return vec4(colored+dst.rgb*(1.0-sa),sa+da*(1.0-sa)); }
    let cs = select(vec3(0.0),colored/max(sa,0.000001),sa>0.0);
    let cb = select(vec3(0.0),dst.rgb/max(da,0.000001),da>0.0);
    let mixed = color_blend(cb,cs,mode);
    let cr = (1.0-da)*cs+da*mixed;
    var co = sa*cr+dst.rgb*(1.0-sa);
    var ao = sa+da*(1.0-sa);
    switch params.modes.y {
        case 1u: {co=sa*cr*da+dst.rgb*(1.0-sa);ao=da;}
        case 2u: {ao=da*(1.0-sa);co=cb*ao;}
        case 3u: {co=sa*cr+max(da-sa,0.0)*cb;ao=max(sa,da);}
        case 4u: {co=sa*cr+min(da,1.0-sa)*cb;ao=min(sa+da,1.0);}
        default: {}
    }
    return vec4(co,ao);
}
@fragment fn fs_present(input: Output) -> @location(0) vec4<f32> {
    return textureSampleLevel(source,source_sampler,input.uv,0.0);
}
