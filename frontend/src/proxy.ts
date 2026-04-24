import { NextRequest, NextResponse } from "next/server";

// Routes that require authentication
const PROTECTED_PREFIXES = [
  "/dashboard",
  "/generate",
  "/explore",
  "/history",
  "/settings",
];

// Routes that logged-in users should not see
const AUTH_ROUTES = ["/login", "/signup"];

// Cookie name written by login/signup pages
const TOKEN_COOKIE = "sf_access";

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Handle legacy /app/* redirect → strip the /app prefix
  if (pathname.startsWith("/app/")) {
    const stripped = pathname.replace(/^\/app/, "") || "/dashboard";
    const url = request.nextUrl.clone();
    url.pathname = stripped;
    return NextResponse.redirect(url);
  }

  const token = request.cookies.get(TOKEN_COOKIE)?.value;
  const isProtected = PROTECTED_PREFIXES.some((prefix) =>
    pathname.startsWith(prefix)
  );
  const isAuthRoute = AUTH_ROUTES.some((r) => pathname === r);

  // Redirect unauthenticated users away from protected routes
  if (isProtected && !token) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("next", pathname);
    return NextResponse.redirect(url);
  }

  // Redirect authenticated users away from login/signup
  if (isAuthRoute && token) {
    const next = request.nextUrl.searchParams.get("next") ?? "/dashboard";
    const url = request.nextUrl.clone();
    // Only allow internal paths for the `next` param (security)
    url.pathname = next.startsWith("/") && !next.startsWith("//") ? next : "/dashboard";
    url.searchParams.delete("next");
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths EXCEPT:
     * - _next/static  (static files)
     * - _next/image   (image optimization)
     * - favicon.ico
     * - public assets (svg, png, jpg, etc.)
     */
    "/((?!_next/static|_next/image|favicon\\.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|woff2?|ttf|otf)).*)",
  ],
};
