"""GLSL 1.50 shaders for the city: one lit shader (sun + hemisphere ambient + PCF soft shadows + fog + facade windows
that light up at night + ego headlights) and one unlit shader (sky, lamps, HUD-like overlays)."""
from __future__ import annotations

from panda3d.core import Shader

_LIGHT_STRUCT = """
uniform struct {
    vec4 position;
    vec3 color;
    vec3 attenuation;
    vec3 spotDirection;
    float spotCosCutoff;
    float spotExponent;
    sampler2DShadow shadowMap;
    mat4 shadowViewMatrix;
} p3d_LightSource[1];
"""

CITY_VERT = """#version 150
""" + _LIGHT_STRUCT + """
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
uniform float world_uv;                 // > 0: texture coordinates = world xz / world_uv  (seam-free asphalt, ground)
out vec3 vpos;
out vec3 wpos;
out vec3 vnorm;
out vec3 wnorm;
out vec4 shad;
out vec4 vcol;
out vec2 uv;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vpos = vec3(p3d_ModelViewMatrix * p3d_Vertex);
    wpos = vec3(p3d_ModelMatrix * p3d_Vertex);
    vnorm = normalize(p3d_NormalMatrix * p3d_Normal);
    wnorm = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    shad = p3d_LightSource[0].shadowViewMatrix * vec4(vpos, 1.0);
    vcol = p3d_Color;
    uv = world_uv > 0.0 ? wpos.xz / world_uv : p3d_MultiTexCoord0;
}
"""

CITY_FRAG = """#version 150
""" + _LIGHT_STRUCT + """
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform vec3 sun_color;
uniform vec3 amb_sky;
uniform vec3 amb_ground;
uniform vec3 fog_color;
uniform float fog_density;
uniform float night;                    // 0 day .. 1 night
uniform float spec_amt;                 // specular strength of this node
uniform float glass_amt;                // texture alpha is a window mask (facades)
uniform float shadows_on;
uniform float alpha_cut;                // > 0: discard transparent texels (traffic signs)
uniform float use_alpha;                // 1: output vertex alpha (glass, tunnel roof)
uniform float cut_on;                   // tunnel cut-away: hide surface layers around the ego while it is underground
uniform float cut_apply;                // 1 for ground-level layers
uniform vec2 cut_c;
uniform float cut_r;
uniform vec3 head_pos;                  // world-space ego headlight position / direction
uniform vec3 head_dir;
uniform float head_on;
uniform vec3 cam_pos;
in vec3 vpos;
in vec3 wpos;
in vec3 vnorm;
in vec3 wnorm;
in vec4 shad;
in vec4 vcol;
in vec2 uv;
out vec4 p3d_FragColor;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

float shadow_factor(vec4 sc, float ndl) {
    if (shadows_on < 0.5) return 1.0;
    float bias = 0.0007 + 0.0035 * (1.0 - ndl);
    sc.z -= bias * sc.w;
    float s = 0.0;
    const float o = 0.00055;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            s += textureProj(p3d_LightSource[0].shadowMap, sc + vec4(float(i) * o * sc.w, float(j) * o * sc.w, 0.0, 0.0));
    return s / 9.0;
}

void main() {
    vec4 tex = texture(p3d_Texture0, uv);
    if (alpha_cut > 0.5 && tex.a < 0.5) discard;
    if (cut_on > 0.5 && cut_apply > 0.5 && wpos.y > -0.35 && wpos.y < 1.2 && distance(wpos.xz, cut_c) < cut_r) discard;
    vec3 albedo = tex.rgb * vcol.rgb * p3d_ColorScale.rgb;
    vec3 N = normalize(wnorm);
    if (!gl_FrontFacing) N = -N;
    vec3 Lv = normalize(p3d_LightSource[0].position.xyz);        // view space, towards the sun (directional)
    float ndl = max(dot(normalize(gl_FrontFacing ? vnorm : -vnorm), Lv), 0.0);
    float sh = shadow_factor(shad, ndl);

    vec3 ambient = mix(amb_ground, amb_sky, N.y * 0.5 + 0.5);
    vec3 light = ambient + sun_color * ndl * sh;

    vec3 V = normalize(-vpos);
    vec3 H = normalize(Lv + V);
    float spec = pow(max(dot(normalize(gl_FrontFacing ? vnorm : -vnorm), H), 0.0), 48.0) * spec_amt * sh * step(0.001, ndl);

    // ego headlights (night / tunnels)
    if (head_on > 0.0) {
        vec3 d = head_pos - wpos;
        float dist = length(d);
        vec3 Ld = d / max(dist, 0.01);
        float cone = smoothstep(0.80, 0.93, dot(-Ld, head_dir));
        float att = 1.0 / (1.0 + 0.030 * dist + 0.0016 * dist * dist);
        light += vec3(1.0, 0.92, 0.75) * 2.6 * cone * att * max(dot(N, Ld), 0.0) * head_on;
    }
    vec3 col = albedo * light + vec3(spec) * sun_color * 0.6;

    // facades: window mask in the texture alpha -> sky reflection by day, warm light at night
    if (glass_amt > 0.0) {
        float win = tex.a;
        vec3 refl = mix(fog_color, amb_sky * 1.6, 0.5);
        float fres = pow(1.0 - max(dot(normalize(gl_FrontFacing ? vnorm : -vnorm), V), 0.0), 2.0);
        col = mix(col, refl * (0.55 + 0.6 * ndl * sh) + spec * 2.0, win * (0.35 + 0.5 * fres) * (1.0 - night));
        vec2 cell = floor(uv * vec2(1.0, 1.0));
        float lit = step(0.52, hash(cell + vec2(vcol.a * 91.0, vcol.a * 17.0)));
        vec3 warm = mix(vec3(1.0, 0.78, 0.45), vec3(0.65, 0.85, 1.0), step(0.85, hash(cell * 1.7 + vcol.a)));
        col = mix(col, warm * 1.35, win * lit * night);
    }

    // fog
    float dist = length(vpos);
    float f = 1.0 - exp(-pow(dist * fog_density, 1.25));
    col = mix(col, fog_color, clamp(f, 0.0, 1.0));
    p3d_FragColor = vec4(col, use_alpha > 0.5 ? vcol.a * tex.a : 1.0);
}
"""

UNLIT_VERT = """#version 150
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;
out vec4 vcol;
out vec2 uv;
out float vdist;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vcol = p3d_Color;
    uv = p3d_MultiTexCoord0;
    vdist = length(vec3(p3d_ModelViewMatrix * p3d_Vertex));
}
"""

UNLIT_FRAG = """#version 150
uniform sampler2D p3d_Texture0;
uniform vec4 p3d_ColorScale;
uniform vec3 fog_color;
uniform float fog_density;
uniform float use_fog;
uniform float glow;                    // brightness multiplier (lamps at night)
in vec4 vcol;
in vec2 uv;
in float vdist;
out vec4 p3d_FragColor;
void main() {
    vec4 c = texture(p3d_Texture0, uv) * vcol * p3d_ColorScale;
    c.rgb *= glow;
    if (use_fog > 0.5) {
        float f = 1.0 - exp(-pow(vdist * fog_density, 1.25));
        c.rgb = mix(c.rgb, fog_color, clamp(f, 0.0, 1.0));
    }
    p3d_FragColor = c;
}
"""


def city_shader() -> Shader:
    return Shader.make(Shader.SL_GLSL, vertex=CITY_VERT, fragment=CITY_FRAG)


def unlit_shader() -> Shader:
    return Shader.make(Shader.SL_GLSL, vertex=UNLIT_VERT, fragment=UNLIT_FRAG)
