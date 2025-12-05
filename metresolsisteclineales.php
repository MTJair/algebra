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
<section id="tema4">
  <h1>Tema 4 — Métodos de resolución de sistemas de ecuaciones lineales</h1>
  <p>Este tema presenta los métodos más utilizados para resolver sistemas de ecuaciones lineales: Regla de Cramer, Criterio de Rouché–Frobenius, Eliminación de Gauss y las descomposiciones QR y Cholesky. Se incluyen ejemplos matemáticos, interpretaciones intuitivas y código en <code>Python</code> listo para usarse en Spyder.</p>

  <!-- 4.1 ------------------------------------------------------------------------------->
  <article id="4-1">
    <h2>4.1 Regla de Cramer</h2>

    <h4>Definición</h4>
    <p>La <strong>Regla de Cramer</strong> permite resolver sistemas lineales cuadrados de la forma <em>Ax=b</em> cuando <strong>det(A) ≠ 0</strong>.  
      La solución para cada variable es:</p>
      <br>
      <em>\( x_i = \frac{\det(A_i)}{\det(A)} \)</em>,
      <br>
     <p>
      donde \(A_i\) es la matriz A reemplazando la columna i por el vector b.
    </p>
    <br>
    <h4>Ejemplo</h4>
    <br>
    <blockquote>
      \(\begin{cases}
      x + 2y = 5 \\
      3x - y = 4
      \end{cases}\)
    </blockquote>
    <br>
    <h4>Ejemplo en Python (Regla de Cramer)</h4>
    <pre><code class="language-python">
import numpy as np

def cramer(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    detA = np.linalg.det(A)
    if abs(detA) < 1e-12:
        raise ValueError("El sistema no tiene solución única (det(A)=0)")
    n = A.shape[1]
    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA
    return x

# Ejemplo
A = np.array([[1, 2],
              [3, -1]])
b = np.array([5, 4])

sol = cramer(A, b)
print("Solución por Cramer:", sol)
</code></pre>

    <br>
    <h4>Limitaciones</h4>
    <ul>
      <li>Solo aplica a sistemas cuadrados.</li>
      <li>No es eficiente para matrices grandes (calcular determinantes es costoso).</li>
    </ul>
  </article>
  <br>

  <!-- 4.2 ------------------------------------------------------------------------------->
  <article id="4-2">
    <h2>4.2 Criterio de Rouché–Frobenius (clasificación por rangos)</h2>

    <h4>Definición</h4>
    <p>Dado el sistema <em>Ax=b</em>, se consideran sus rangos:</p>
    <ul>
      <li><strong>rank(A)</strong>: dimensión del espacio generado por las filas/columnas.</li>
      <li><strong>rank([A|b])</strong>: rango de la matriz aumentada.</li>
    </ul>

    <p>Entonces:</p>
    <ul>
      <li>Si <strong>rank(A)=rank([A|b])=n</strong> → solución única.</li>
      <li>Si <strong>rank(A)=rank([A|b]) &lt; n</strong> → infinitas soluciones.</li>
      <li>Si <strong>rank(A) &lt; rank([A|b])</strong> → sin solución.</li>
    </ul>
    <br>
    <h4>Ejemplo en Python</h4>
    <pre><code class="language-python">
import numpy as np

def clasificar_sistema(A, b):
    A = np.array(A, float)
    b = np.array(b, float)
    rA = np.linalg.matrix_rank(A)
    rAb = np.linalg.matrix_rank(np.column_stack([A, b]))

    print("rank(A) =", rA, " | rank([A|b]) =", rAb)

    n = A.shape[1]
    if rA == rAb == n:
        print("→ Solución única")
    elif rA == rAb < n:
        print("→ Infinitas soluciones")
    else:
        print("→ Sistema incompatible (sin solución)")

# Ejemplo
A = np.array([[1, -1], [2, -2]])
b = np.array([1, 3])
clasificar_sistema(A, b)
</code></pre>
    <br>
    <h4>Interpretación geométrica</h4>
    <p>El criterio generaliza la idea de rectas y planos:  
      <ul>
        <li>intersección en un punto → única solución,</li>
        <li>coincidencia → infinitas soluciones,</li>
        <li>paralelismo → sin solución.</li>
      </ul>
    </p>
  </article>

  <!-- 4.3 ------------------------------------------------------------------------------->
  <article id="4-3">
    <h2>4.3 Algoritmo de Gauss</h2>

    <h4>Idea general</h4>
    <p>Consiste en transformar la matriz aumentada <strong>[A|b]</strong> mediante operaciones elementales de fila hasta obtener una forma escalonada, desde la cual se resuelve por sustitución hacia atrás.</p>

    <h4>Pasos</h4>
    <br>
    <ol>
      <li>Usar pivotes para eliminar coeficientes por debajo.</li>
      <li>Llevar la matriz a forma escalonada.</li>
      <li>Realizar sustitución hacia atrás.</li>
    </ol>
    <br>
    <h4>Implementación pedagógica en Python</h4>
    <pre><code class="language-python">
import numpy as np

def gauss(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape

    # Matriz aumentada
    M = np.hstack([A, b.reshape(-1,1)])

    for i in range(n):
        # 1. Seleccionar pivote (pivoteo parcial)
        pivot = i + np.argmax(np.abs(M[i:, i]))
        if abs(M[pivot, i]) < 1e-12:
            continue  # columna sin pivote

        # Intercambio de filas
        M[[i, pivot]] = M[[pivot, i]]

        # 2. Normalizar pivote
        M[i] = M[i] / M[i, i]

        # 3. Eliminar hacia abajo
        for j in range(i+1, m):
            M[j] -= M[j, i] * M[i]

    return M

# Ejemplo
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], float)
b = np.array([8, -11, -3], float)

M = gauss(A, b)
print("Matriz escalonada:")
print(M)
</code></pre>

    <br>
    <h4>Implementación de sustitución hacia atrás</h4>
    <pre><code class="language-python">
def back_substitution(M):
    M = np.array(M, float)
    m, n1 = M.shape
    n = n1 - 1

    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])
    return x

# Solución del sistema
# x = back_substitution(M)
# print("Solución:", x)
</code></pre>
  </article>
  <br>
  <!-- 4.4 ------------------------------------------------------------------------------->
  <article id="4-4">
    <h2>4.4 Descomposiciones QR y de Cholesky</h2>

    <h4>Descomposición QR</h4>
    <p>Divide la matriz A como:
    <br>
    <em>A = QR</em>,
    </p>
    <ul>
      <li><strong>Q</strong> es ortogonal (QᵀQ = I),</li>
      <li><strong>R</strong> es triangular superior.</li>
    </ul>

    <p>La solución de <em>Ax=b</em> se obtiene resolviendo:</p>
    <blockquote>R x = Qᵀ b</blockquote>
    <br>

    <h4>Ejemplo en Python (QR)</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[1,1],[2,-1]], float)
b = np.array([3,0], float)

Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)

print("Solución (QR):", x)
</code></pre>

    <br>
    <h4>Descomposición de Cholesky</h4>
    <p>Aplica solo a matrices <strong>simétricas y definidas positivas</strong>.  
      Se factoriza:
      <br>
      <em>A = L Lᵀ</em>
    </p>

    <h4>Ejemplo en Python (Cholesky)</h4>
    <pre><code class="language-python">
import numpy as np

A = np.array([[4, 2],
              [2, 3]], float)
b = np.array([6, 5], float)

L = np.linalg.cholesky(A)

# resolver Ly = b
y = np.linalg.solve(L, b)

# resolver Lᵀx = y
x = np.linalg.solve(L.T, y)

print("Solución (Cholesky):", x)
</code></pre>

    <br>
    <h4>Ventajas</h4>
    <br>    
    <ul>
      <li>Cholesky: muy eficiente (la mitad del costo de una LU).</li>
      <li>QR: muy estable numéricamente (preferido en mínimos cuadrados).</li>
    </ul>
  </article>

  <br>

  <footer>
    <h3>Actividades y tareas</h3>
    <ol>
      <li>Programa tu propia descomposición QR mediante Gram–Schmidt clásico y modificado.</li>
      <li>Simula sistemas aleatorios y compara la estabilidad de Gauss vs QR.</li>
      <li>Implementa Cholesky para matrices 3×3 de forma manual y verifica sus propiedades.</li>
    </ol>

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