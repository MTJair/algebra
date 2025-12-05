<?php
session_start();
if (!isset($_SESSION['name']) || $_SESSION['module'] !== 'algebra') {
    header('Location: http://www.quantcodehub.com/clases.php');
    exit();}
?>
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">

    <!--Page Title-->
    <title>Math!</title>

    <!--Meta Keywords and Description-->
    <meta name="keywords" content="">
    <meta name="description" content="">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>

    <!--Favicon-->
    <link rel="shortcut icon" href="images/math.png" title="Favicon"/>

    <!-- Main CSS Files -->
    <link rel="stylesheet" href="css/style.css">

    <!-- Namari Color CSS -->
    <link rel="stylesheet" href="css/namari-color.css">

    <!--Icon Fonts - Font Awesome Icons-->
    <link rel="stylesheet" href="css/font-awesome.min.css">

    <!-- Animate CSS-->
    <link href="css/animate.css" rel="stylesheet" type="text/css">

    <!--Google Webfonts-->
    <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,300,600,700,800' rel='stylesheet' type='text/css'>

    <style type="text/css">
    .text-center {
        text-align: center;}
    .container {    
    display: flex; !important;
    justify-content: center; !important;}
    .col-2.icon-block {
      border: 2px solid black; /* Establece un borde sólido de 2px en color negro */
      border-radius: 5px; /* Agrega esquinas redondeadas */
      padding: 10px; /* Añade un relleno interno de 10px */
      cursor: pointer; /* Cambia el cursor al pasar sobre el elemento */
      transition: all 0.3s ease; /* Agrega una transición suave a todos los cambios */
    }
    /* Cambia el color del borde al pasar el ratón sobre el elemento */
    .col-2.icon-block:hover {
      border-color: gray;
    }
    #justificado {text-align: justify;}
    #justificado2{            
        text-align: justify;
        margin-left: 20px; /* Margen a la izquierda */
        margin-right: 20px; /* Margen a la derecha */
    }    
    #centrado {text-align: center;}
    #derecha {text-align: right;}
    }
        .button-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
        } 
        .subtema-button {
            background-color: #82E0AA;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .subtema-button:hover {
            background-color: #322E2D;
        }
        .subtema-button-2 {
            background-color: #636260;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .subtema-button-2:hover {
            background-color: #322E2D;
        }
        #justificado a {
            color: blue;
            font-weight: bold;
            white-space: nowrap;  /* Evita el salto de línea */
            text-transform: none; /* Evita que el texto esté en mayúsculas */
            display: inline;      /* Asegura que el enlace se comporte como un elemento en línea */
        }
    </style>

    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

</head>
<body>

<!-- Preloader -->
<div id="preloader">
    <div id="status" class="la-ball-triangle-path">
        <div></div>
        <div></div>
        <div></div>
    </div>
</div>
<!--End of Preloader-->

<div class="page-border" data-wow-duration="0.7s" data-wow-delay="0.2s">
    <div class="top-border wow fadeInDown animated" style="visibility: visible; animation-name: fadeInDown;"></div>
    <div class="right-border wow fadeInRight animated" style="visibility: visible; animation-name: fadeInRight;"></div>
    <div class="bottom-border wow fadeInUp animated" style="visibility: visible; animation-name: fadeInUp;"></div>
    <div class="left-border wow fadeInLeft animated" style="visibility: visible; animation-name: fadeInLeft;"></div>
</div>

<div id="wrapper">

    <header id="banner" class="scrollto clearfix" data-enllax-ratio=".5">
        <div id="header" class="nav-collapse">
            <div class="row clearfix">
                <div class="col-1">

                    <!--Main Navigation-->
                    <nav id="nav-main">
                        <ul>
                            <li>
                                <a href="/clases.php">Inicio</a>
                            </li>
                            <li>
                                <a href="#clients">Álgebra</a>
                            </li>
                        </ul>
                    </nav>
                    <!--End of Main Navigation-->

                    <div id="nav-trigger"><span></span></div>
                    <nav id="nav-mobile"></nav>

                </div>
            </div>
        </div><!--End of Header-->

    </header>

    <!--Main Content Area-->
    <main id="content">

        <!--Calculadoras-->
        <section id="clients" class="scrollto clearfix">
            <div class="row clearfix">
                <div class="col-3">

                    <div class="section-heading">
                        <h3>Matemáticas</h3>
                        <h2 class="section-title">Álgebra</h2>
                        <div class="button-container">
                            <a href="" class="subtema-button">Matrices y determinantes</a>
                            <a href="/ecuacionesalgebraicas.php" class="subtema-button">Ecuaciones algebraicas</a>
                            <a href="/sistemaseclineales.php" class="subtema-button">Sistemas de ecuaciones lineales</a> 
                            <a href="/metresolsisteclineales.php" class="subtema-button">Métodos de resolución de sistemas de ecuaciones lineales</a>
                            <a href="/espaciosvectoriales.php" class="subtema-button">Espacios vectoriales</a>
                            <a href="/aplicacioneslineales.php" class="subtema-button">Aplicaciones lineales</a>
                            <a href="/pca.php" class="subtema-button">PCA: análisis de componentes principales</a>
                            <a href="/svd.php" class="subtema-button">SVD: descomposición en valores singulares</a>
                            <a href="/algebra_evaluaciones.php" class="subtema-button-2">Evaluaciones</a>
                        </div>
                    </div>

                </div>

            <div class="col-2-3">

<!-- Copia todo esto dentro de la etiqueta <body> de tu página HTML -->
<section id="tema-ecuaciones">
  <h1>Tema 2 — Ecuaciones algebraicas</h1>
  <p>En este tema estudiaremos la resolución y factorización de polinomios y ecuaciones algebraicas. Contiene explicaciones formales y didácticas, ejemplos resueltos, código en <code>Python</code> y gráficos ilustrativos.</p>

  <article id="2-1">
    <h2>2.1 Resolución de ecuaciones algebraicas de grado 2 y 3</h2>

    <h3>Ecuación cuadrática (grado 2)</h3>
    <p>Una ecuación cuadrática general se escribe
      <br><em>ax<sup>2</sup> + bx + c = 0</em>, con <em>a ≠ 0</em>.
      La <strong>fórmula cuadrática</strong> da las soluciones:
    </p>
    <blockquote>
      x = \( \dfrac{-b \pm \sqrt{b^2 - 4ac}}{2a} \)
    </blockquote>
    <p>Donde el discriminante \(\Delta = b^2 - 4ac\) determina la naturaleza de las raíces:
      <ul>
        <li>\(\Delta > 0\): dos raíces reales distintas.</li>
        <li>\(\Delta = 0\): una raíz real doble (multiplicidad 2).</li>
        <li>\(\Delta < 0\): dos raíces complejas conjugadas.</li>
      </ul>
    </p>

    <h4>Ejemplo y código (Python)</h4>
    <br>
    <pre><code class="language-python"># Resolver ax^2 + bx + c = 0 (numérico) - Spyder
import math

def solve_quadratic(a, b, c):
    if a == 0:
        raise ValueError('No es cuadrática (a = 0)')
    delta = b*b - 4*a*c
    if delta >= 0:
        x1 = (-b + math.sqrt(delta)) / (2*a)
        x2 = (-b - math.sqrt(delta)) / (2*a)
    else:
        # raíces complejas
        from cmath import sqrt
        x1 = (-b + sqrt(delta)) / (2*a)
        x2 = (-b - sqrt(delta)) / (2*a)
    return x1, x2

# Ejemplo
a, b, c = 1, -3, 2  # x^2 - 3x + 2 = 0 -> raíces 1 y 2
r1, r2 = solve_quadratic(a, b, c)
print('Raíces:', r1, r2)
</code></pre>
    <br>
    <h4>Gráfica de la cuadrática</h4>
    <br>
    <pre><code class="language-python"># Graficar la función cuadrática y marcar raíces
import numpy as np
import matplotlib.pyplot as plt

a, b, c = 1, -3, 2
x = np.linspace(-1, 4, 400)
fx = a*x**2 + b*x + c
plt.figure()
plt.plot(x, fx, label=f'{a}x^2 + {b}x + {c}')
roots = solve_quadratic(a, b, c)
for r in roots:
    if abs(r.imag) < 1e-8:  # raíz real
        plt.scatter(r.real, 0, color='red')
plt.axhline(0, color='gray')
plt.grid(True)
plt.legend()
plt.title('Parábola y sus raíces')
plt.show()
</code></pre>

    <br>
    <h3>Ecuación cúbica (grado 3)</h3>
    <p>Una ecuación cúbica general se escribe
      <br><em>ax<sup>3</sup> + bx<sup>2</sup> + cx + d = 0</em>, con <em>a ≠ 0</em>.
    </p>
    <p>Existe una fórmula cerrada (método de <strong>Cardano</strong>) para resolver cúbicas, pero es larga y delicada (comporta raíces cúbicas complejas intermedias). En la práctica, para propósitos pedagógicos y numéricos, se suele usar métodos numéricos o bibliotecas que devuelven las tres raíces (reales o complejas).</p>

    <h4>Idea (esquema de Cardano)</h4>
    <br>
    <ol>
      <li>Se transforma la cúbica general en una <em>cúbica deprimida</em> mediante el cambio de variable \(x = y - b/(3a)\).</li>
      <li>Se resuelve la forma deprimida \(y^3 + py + q = 0\) con fórmulas que involucran raíces cúbicas de expresiones con \(q/2 \pm \sqrt{(q/2)^2 + (p/3)^3}\).</li>
    </ol>
    <p>Debido a la complejidad (casos con números complejos incluso cuando hay tres raíces reales — la llamada <em>caso casus irreducibilis</em>), usaremos herramientas computacionales para mostrar ejemplos.</p>

    <h4>Ejemplo y código (Python — numérico y simbólico)</h4>
    <br>
    <pre><code class="language-python"># Resolver una cúbica usando numpy.roots y sympy (si está disponible)
import numpy as np

# coeficientes para a x^3 + b x^2 + c x + d
coeffs = [1, -6, 11, -6]  # x^3 -6x^2 +11x -6 -> raíces 1,2,3
roots_numeric = np.roots(coeffs)
print('Raíces (numpy):', roots_numeric)

# Con sympy (exacto simbólico) si está instalado:
try:
    import sympy as sp
    x = sp.symbols('x')
    poly = sum(coeffs[i]*x**(3-i) for i in range(4))
    roots_sym = sp.solve(sp.Eq(poly,0), x)
    print('Raíces (sympy):', roots_sym)
except Exception as e:
    print('sympy no disponible o error:', e)
</code></pre>

    <br>
    <h4>Gráfica de una cúbica</h4>
    <br>
    <pre><code class="language-python"># Graficar la cúbica y marcar raíces reales
import numpy as np
import matplotlib.pyplot as plt

coeffs = [1, -6, 11, -6]
a, b, c, d = coeffs
x = np.linspace(-1, 4, 400)
fx = a*x**3 + b*x**2 + c*x + d
plt.figure()
plt.plot(x, fx, label='cúbica')
roots = np.roots(coeffs)
for r in roots:
    if abs(r.imag) < 1e-8:
        plt.scatter(r.real, 0, color='red')
plt.axhline(0, color='gray')
plt.grid(True)
plt.title('Cúbica y sus raíces reales')
plt.show()
</code></pre>
  </article>

  <article id="2-2">
    <h2>2.2 Cálculo de soluciones mediante la Regla de Ruffini (división sintética)</h2>

    <h3>Descripción</h3>
    <p>La <strong>Regla de Ruffini</strong> (división sintética) es un algoritmo eficiente para dividir un polinomio por un binomio de la forma (x - r). Permite también comprobar raíces racionales y reducir el grado del polinomio cuando encontramos una raíz.</p>

    <h4>Algoritmo (esquema)</h4>
    <ol>
      <li>Escribe los coeficientes del polinomio en orden de grados descendentes.</li>
      <li>Lleva el primer coeficiente hacia abajo; multiplícalo por r y súmalo al siguiente coeficiente; repite.</li>
      <li>El último número obtenido es el resto; si es 0, r es raíz y los números anteriores son los coeficientes del cociente.</li>
    </ol>

    <br>
    <h4>Ejemplo y código (Python)</h4>
    <br>
    <pre><code class="language-python"># Implementación de Ruffini (división sintética)
def ruffini_division(coeffs, r):
    """Divide el polinomio con coeficientes 'coeffs' por (x - r).
    coeffs: lista [a_n, a_{n-1}, ..., a_0]
    Retorna (cociente_coeffs, resto)
    """
    n = len(coeffs)
    out = [coeffs[0]]  # primer coeficiente del cociente
    for i in range(1, n):
        out.append(coeffs[i] + r * out[-1])
    # out contiene coeficientes del cociente y el resto al final
    return out[:-1], out[-1]

# Ejemplo: dividir x^3 - 6x^2 + 11x -6 entre (x-1)
coeffs = [1, -6, 11, -6]
cociente, resto = ruffini_division(coeffs, 1)
print('Cociente (coef):', cociente)
print('Resto:', resto)  # si resto==0, r=1 es raíz
</code></pre>

    <br>
    <h4>Uso pedagógico</h4>
    <p>Si encontramos una raíz racional r (por ejemplo usando el Teorema del Residuo o el Teorema de la Raíz Racional), aplicamos Ruffini para reducir el polinomio y repetimos hasta factorizar completamente.</p>
  </article>

  <article id="2-3">
    <h2>2.3 Factorización de polinomios sobre ℤ, ℚ, ℝ y ℂ</h2>

    <h3>Conceptos esenciales</h3>
    <ul>
      <li><strong>Factorización sobre ℤ (enteros):</strong> escribir el polinomio como producto de polinomios con coeficientes enteros y de grado menor; se suele usar el Teorema de Gauss y el Teorema de la raíz racional para buscar factores lineales.</li>
      <li><strong>Factorización sobre ℚ (racionales):</strong> similar a ℤ: si un polinomio es reducible sobre ℚ se puede escribir con factores en ℚ[x].</li>
      <li><strong>Factorización sobre ℝ (reales):</strong> todo polinomio real se factoriza en factores lineales y factores cuadráticos irreducibles (con discriminante negativo) con coeficientes reales.</li>
      <li><strong>Factorización sobre ℂ (complejos):</strong> por el Teorema Fundamental del Álgebra, todo polinomio de grado n se factoriza completamente en n factores lineales en ℂ[x].</li>
    </ul>

    <br>
    <h4>Ejemplos y código (Python con sympy y numpy)</h4>
    <br>
    <pre><code class="language-python"># Factorización con sympy (si está disponible)
try:
    import sympy as sp
    x = sp.symbols('x')

    # Ejemplo 1: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    poly1 = x**3 - 6*x**2 + 11*x - 6
    print('Factorización exacta (sympy):', sp.factor(poly1))

    # Ejemplo 2: x^4 - 1 (factoriza sobre Z y Q)
    poly2 = x**4 - 1
    print('x^4 - 1 =', sp.factor(poly2))

    # Factorización sobre reales (force domain='RR') y sobre complejos ('CC')
    print('Factor (RR):', sp.factor(poly2, modulus=None))
    print('Factor over complex:', sp.factor(poly2, gaussian=True))

except Exception as e:
    print('sympy no disponible:', e)

# Si sympy no está, podemos mostrar raíces numéricas con numpy
import numpy as np
coeffs = [1, -6, 11, -6]
roots = np.roots(coeffs)
print('Raíces numéricas (numpy):', roots)
</code></pre>

    <br>
    <h4>Teorema de la raíz racional (práctico)</h4>
    <p>Si un polinomio con coeficientes enteros \(a_n x^n + ... + a_0\) tiene una raíz racional \(p/q\) (en fracción irreducible), entonces \(p\) divide a \(a_0\) y \(q\) divide a \(a_n\). Esto reduce las candidatas a probar con Ruffini.</p>

    <h4>Ejemplo: factorización por pasos</h4>
    <br>
    <ol>
      <li>Aplicar el teorema de la raíz racional para generar candidatos.</li>
      <li>Probar candidatos con Ruffini hasta encontrar una raíz r (resto 0).</li>
      <li>Reducir el polinomio y repetir.</li>
    </ol>

    <br>
    <h4>Factorización sobre ℝ vs ℂ</h4>
    <p>Si un polinomio real tiene raíces complejas no reales, éstas aparecen en pares conjugados; así, sobre ℝ podemos agrupar esos pares en factores cuadráticos con coeficientes reales.</p>
  </article>

  <article id="graficas-ejemplos">
    <h2>Visualizaciones y ejercicios prácticos</h2>

    <h4>Graficar un polinomio y marcar raíces (combinando numpy y sympy)</h4>
    <br>
    <pre><code class="language-python"># Graficar polinomio y marcar todas sus raíces (reales y proyectadas)
import numpy as np
import matplotlib.pyplot as plt

# Coeficientes de ejemplo: x^3 - 6x^2 + 11x - 6
coeffs = [1, -6, 11, -6]
roots = np.roots(coeffs)

x = np.linspace(min(roots.real)-2, max(roots.real)+2, 400)
fx = np.polyval(coeffs, x)

plt.figure()
plt.plot(x, fx, label='polinomio')
plt.axhline(0, color='gray')
# marcar raíces reales
for r in roots:
    if abs(r.imag) < 1e-8:
        plt.scatter(r.real, 0, color='red')
plt.title('Polinomio y sus raíces reales (si las hay)')
plt.grid(True)
plt.show()
</code></pre>

    <br>
    <h4>Ejercicios recomendados</h4>

    <ol>
      <li>Resuelve y grafica varias cuadráticas con diferentes discriminantes; comenta el comportamiento de la parábola.</li>
      <li>Aplica Ruffini para factorizar x^3 - 6x^2 + 11x - 6 y comprueba los factores.</li>
      <li>Usa sympy para factorizar x^4 + 2x^2 + 1 y compara con el resultado manual.</li>
      <li>Encuentra y grafica las raíces de x^3 - 3x + 1; discute si son reales o complejas.</li>
    </ol>
  </article>

  <footer>
    <p>— <strong>Prof. Eder Díaz</strong></p>
  </footer>
</section>


            </div>

            </div>
        </section>
        <!--End of Calculadoras-->

    </main>
    <!--End Main Content Area-->

</div>

<!-- Include JavaScript resources -->
<script src="js/jquery.1.8.3.min.js"></script>
<script src="js/wow.min.js"></script>
<script src="js/featherlight.min.js"></script>
<script src="js/featherlight.gallery.min.js"></script>
<script src="js/jquery.enllax.min.js"></script>
<script src="js/jquery.scrollUp.min.js"></script>
<script src="js/jquery.easing.min.js"></script>
<script src="js/jquery.stickyNavbar.min.js"></script>
<script src="js/jquery.waypoints.min.js"></script>
<script src="js/images-loaded.min.js"></script>
<script src="js/lightbox.min.js"></script>
<script src="js/site.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://kit.fontawesome.com/30c014f357.js" crossorigin="anonymous"></script>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

</body>
</html>