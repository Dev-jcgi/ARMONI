// Antepone la ruta base del sitio ('/' en armoni.bio, '/ARMONI' en jcgi.lat/ARMONI/).
export const u = (p: string) => import.meta.env.BASE_URL.replace(/\/$/, '') + p;
